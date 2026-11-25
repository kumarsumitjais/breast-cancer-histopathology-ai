# Frontend

Single-page upload UI: React + Vite + TypeScript, plain CSS.

Upload one or more images → each is sent to `POST /predict` (one request per image, so one bad file doesn't fail the rest) → class, malignant probability and the 0.88 threshold are shown per image. "Show Grad-CAM heatmap" calls `POST /explain` on demand.

## Run

Start the backend first (see root README), then:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The dev server must stay on port 5173 — it's the origin the backend's CORS config allows.

API URL defaults to `http://127.0.0.1:8000`; override with `VITE_API_BASE_URL` in `frontend/.env` (see `.env.example`).
