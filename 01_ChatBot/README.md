# Subsystem A — ShopSphere E-commerce Chatbot

React (Vite) frontend + FastAPI backend + Ollama LLM. The "app under test" for the DeepEval framework.

## Ports
| Service | Port |
|--------|------|
| FastAPI backend | 8201 |
| Vite dev server | 5173 |

## Run

### 1. Run Ollama Locally
Ensure Ollama is running and has the model loaded (e.g. `gemma3:4b` or `qwen3:4b`):
```bash
ollama run gemma3:4b
```

### 2. Run Backend
Create a `.env` file under `backend` or configure environment variables:
```bash
# Terminal 1 — backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8201
```

Default Environment Variables:
- `OLLAMA_BASE_URL` = `http://localhost:11434/v1`
- `CHATBOT_MODEL` = `gemma3:4b` (or `qwen3:4b`)

### 3. Run Frontend
```bash
# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

## API
- `GET /health` — status + active model
- `POST /chat` — `{message, history?}` → `{reply, model, mode}`
