# Falco Service + Web Setup

## 1) Backend (FastAPI)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API server from project root:

```bash
uvicorn service.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:

- `GET /api/v1/health`
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream` (SSE)
- `POST /api/v1/rag/search`
- `POST /api/v1/rag/index`

## 2) Frontend (Next.js)

```bash
cd web
npm install
npm run dev
```

Default web URL: `http://127.0.0.1:1357`

Environment variable (optional):

```bash
NEXT_PUBLIC_FALCO_API_BASE=http://127.0.0.1:8000
```

## 3) Notes

- Frontend default API base is `http://127.0.0.1:8000`.
- `thread_id` controls conversation continuity and memory scope.
- Falco core code remains in `src/`.
