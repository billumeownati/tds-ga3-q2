"""Local protocol/storage smoke tests; they use a fake AI and do not call a model."""
import json
import os
import re
import tempfile

# Use a disposable local database before importing main (which initializes tables).
db_path = tempfile.mktemp(prefix="invoice-agent-smoke-", suffix=".db")
os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
os.environ["A2A_BASE_URL"] = "http://127.0.0.1:8000/"

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402


def fake_model(_policy_context, requested):
    result = {}
    for item in requested:
        package = item["package"]
        text = json.dumps(package, ensure_ascii=False)
        refs = list(dict.fromkeys(m.group(0) for m in re.finditer(r"\[[^\[\]\r\n]{1,120}\]", text)))
        result[item["packageId"]] = {
            "action": "hold_invoice",
            "facts": {"vendorName": "Example Vendor", "invoiceNumber": "INV-TEST", "amountMinor": 12500, "currency": "INR"},
            "evidenceRefs": refs[:3],
            "rationale": f"Action hold_invoice because verification is required. The decisive evidence is {refs[0]}, {refs[1]}, and {refs[2]}; confirm the source records before releasing payment.",
        }
    return result


main.model_decisions = fake_model
client = TestClient(main.app)
headers = {
    "Authorization": "Bearer smoke-user-one",
    "A2A-Version": "1.0",
    "Content-Type": "application/a2a+json",
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


try:
    card = client.get("/.well-known/agent-card.json")
    check(card.status_code == 200, "Agent Card should be public")
    check(card.json()["supportedInterfaces"][0]["protocolVersion"] == "1.0", "A2A version missing")

    packages = [{
        "packageId": f"smoke-pkg-{i}",
        "documents": [f"Invoice evidence [INV-{i}]", f"Policy evidence [POL-{i}]", f"Ledger evidence [LED-{i}]"],
    } for i in range(2)]
    first_message = {
        "messageId": "smoke-message-first",
        "role": "ROLE_USER",
        "parts": [{"mediaType": main.CLAIM_MIME, "data": {"batchId": "smoke-batch", "policyRevision": "test", "packages": packages}}],
    }
    first = client.post("/message:send", headers=headers, json={"message": first_message, "configuration": {"returnImmediately": False}})
    check(first.status_code == 200, f"Initial request failed: {first.text}")
    task = first.json()["task"]
    check(task["status"]["state"] == "TASK_STATE_INPUT_REQUIRED", "Task must await grader results")
    check(len(task["history"]) == 1 and len(task["artifacts"]) == 1, "Initial history/proposal artifact wrong")

    replay = client.post("/message:send", headers=headers, json={"message": first_message, "configuration": {"returnImmediately": True}})
    check(replay.status_code == 200 and replay.json()["task"]["id"] == task["id"], "Configuration-insensitive replay failed")

    changed = dict(first_message)
    changed["parts"] = []
    conflict = client.post("/message:send", headers=headers, json={"message": changed})
    check(conflict.status_code == 409 and conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT", "Message ID conflict not detected")

    proposal_data = task["artifacts"][0]["parts"][0]["data"]
    results = [{
        "packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
        "outcome": "ACCEPTED", "receiptNonce": f"synthetic-nonce-{p['packageId']}"
    } for p in proposal_data["proposals"]]
    continuation = {
        "messageId": "smoke-message-results", "taskId": task["id"], "contextId": task["contextId"],
        "role": "ROLE_USER",
        "parts": [{"mediaType": main.RESULT_MIME, "data": {"batchId": "smoke-batch", "results": results}}],
    }
    completed_response = client.post("/message:send", headers=headers, json={"message": continuation})
    check(completed_response.status_code == 200, f"Continuation failed: {completed_response.text}")
    completed = completed_response.json()["task"]
    check(completed["status"]["state"] == "TASK_STATE_COMPLETED", "Continuation should complete the Task")
    check(len(completed["history"]) == 2 and len(completed["artifacts"]) == 2, "Final history/artifacts wrong")
    executions = completed["artifacts"][1]["parts"][0]["data"]["executions"]
    check(len(executions) == 2, "Accepted results should become executions")

    terminal_replay = client.post("/message:send", headers=headers, json={"message": continuation})
    check(terminal_replay.status_code == 200 and terminal_replay.json()["task"]["status"]["state"] == "TASK_STATE_COMPLETED", "Terminal replay failed")

    outsider_headers = {"Authorization": "Bearer smoke-user-two", "A2A-Version": "1.0"}
    outsider_read = client.get(f"/tasks/{task['id']}", headers=outsider_headers)
    outsider_list = client.get("/tasks", headers=outsider_headers)
    check(outsider_read.status_code == 404, "Cross-user read must be hidden")
    check(outsider_list.status_code == 200 and outsider_list.json() == {"tasks": []}, "Task list must be owner-scoped")

    cancel_message = {
        "messageId": "smoke-message-cancel", "role": "ROLE_USER",
        "parts": [{"mediaType": main.CLAIM_MIME, "data": {"batchId": "smoke-cancel-batch", "policyRevision": "test", "packages": [{"packageId": "smoke-cancel-pkg", "documents": ["[A]", "[B]", "[C]"]}]}}],
    }
    pending_response = client.post("/message:send", headers=headers, json={"message": cancel_message})
    check(pending_response.status_code == 200, f"Cancel setup failed: {pending_response.text}")
    pending = pending_response.json()["task"]
    canceled_response = client.post(f"/tasks/{pending['id']}:cancel", headers=headers, json={})
    check(canceled_response.status_code == 200, "Cancellation failed")
    canceled = canceled_response.json()
    check(canceled["status"]["state"] == "TASK_STATE_CANCELED", "Task was not canceled")
    check(not any(part.get("mediaType") == main.RECEIPT_MIME for art in canceled["artifacts"] for part in art["parts"]), "Canceled task must not contain receipts")

    print("PASS: Agent Card, proposals, input-required state, idempotency, conflicts, receipts, terminal replay, user isolation, and cancellation")
finally:
    client.close()
    engine = getattr(main, "engine", None)
    if engine is not None:
        engine.dispose()
    try:
        os.remove(db_path)
    except OSError:
        pass
