# A2A Invoice Action Agent

A FastAPI implementation of the assignment's A2A 1.0 HTTP+JSON contract.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
# edit .env for your local Ollama settings, then:
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

This local configuration expects Ollama's OpenAI-compatible endpoint at `http://127.0.0.1:11434/v1` with model `qwen3:4b`. Run Ollama separately and pull the model first. On Windows PowerShell, use `$env:NAME="value"`; on Linux/macOS, use `export NAME="value"`.

Run `python smoke_test.py` to check the protocol/storage lifecycle using a fake model (this does not test invoice-decision accuracy). Visit `http://127.0.0.1:8000/.well-known/agent-card.json` to inspect the public Agent Card. Protected requests need headers:

```text
Authorization: Bearer test-user-a
A2A-Version: 1.0
Content-Type: application/a2a+json
```

## Hosted deployment (Render + PostgreSQL + hosted AI)

1. Push the contents of this folder to a GitHub repository. Do not commit `.env` or any key.
2. In Render, create a **Web Service** connected to that repository. Use Build Command `pip install -r requirements.txt` and Start Command `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`.
3. After Render provides the service's public URL, set `A2A_BASE_URL` to the exact base URL you plan to submit. For example, `https://your-service.onrender.com/a2a/`. If you use `/a2a/`, the API paths are `/a2a/message:send`, `/a2a/tasks`, etc.; the Agent Card stays at the origin-level `/.well-known/agent-card.json`.
4. Configure a hosted OpenAI-compatible AI API in Render's Environment settings. Example using the Gemini OpenAI-compatible endpoint (use a model currently available to your account):

   ```text
   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
   LLM_MODEL=gemini-3.8-flash
   LLM_API_KEY=your_key_from_Google_AI_Studio
   ```

   The key belongs only in Render's secret environment settings, never in source code. The local default (`127.0.0.1:11434`) will not reach Ollama running on your laptop after you deploy the app.
5. Use a durable PostgreSQL database. In Supabase, open **Connect**, copy a PostgreSQL connection string suitable for a persistent backend, and set it as Render's `DATABASE_URL`. Replace the password placeholder and URL-encode reserved password characters. The app converts a normal `postgresql://` string to the SQLAlchemy `postgresql+psycopg://` form. Keep SSL enabled (`sslmode=require`) where the connection string/provider calls for it.
6. Deploy and verify these two public URLs in a browser: `https://your-service.onrender.com/healthz` and `https://your-service.onrender.com/.well-known/agent-card.json`. The health route must return `{"status":"ok"}` and the card must show your exact base URL. Verify no URL redirect is needed.
7. Make one authenticated test request to the API only after configuring the AI and database. Use a small synthetic package; confirm the response is `TASK_STATE_INPUT_REQUIRED`. Do not fabricate a grader receipt—only the grader's continuation can complete the task.

Render's official FastAPI guide documents the build/start commands and public service URL. Supabase's database guide explains where to copy the correct connection string; select based on the backend's network requirements. See their current docs: https://render.com/docs/deploy-fastapi and https://supabase.com/docs/guides/database/connecting-to-postgres.

## Important design choices

- Each bearer token is an isolated principal. Task reads, lists, continuations, and cancellation are owner-scoped.
- The semantic message hash covers the `message` object only (sorted compact JSON); configuration is excluded.
- Every first-turn batch generates proposals and stays in `TASK_STATE_INPUT_REQUIRED`. No business action is executed at proposal time.
- Only a matching result continuation can produce accepted receipt executions and transition the task to `TASK_STATE_COMPLETED`.
- Accepted results are copied from the grader-provided receipt nonce; rejected results are never included in executions.
- Decision cache entries are keyed by the SHA-256 hash of canonical package content. `actionId` values are newly generated for each task.
- Keep one Uvicorn worker because the in-process lock coordinates concurrent message and cancel/result requests.
