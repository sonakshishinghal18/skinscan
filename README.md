# LUMIS — Skin Analysis

Take a selfie. Get a full skin score across five areas, plus tips on exactly what to fix.

## Project structure

```
lumis/
├── index.html          # Frontend — host on GitHub Pages or any static host
├── app.py              # Python Flask backend
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment config
├── .env.example        # Environment variable template
└── .gitignore
```

## Deploy backend to Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo
3. Render auto-detects `render.yaml` and configures the service
4. In the Render dashboard → **Environment** → add:
   - `ANTHROPIC_API_KEY` = `sk-ant-...`
5. Deploy — your backend URL will be `https://<your-service-name>.onrender.com`

## Connect frontend to backend

In `index.html`, update this line to your Render service URL:

```js
const API_BASE = 'https://your-service-name.onrender.com';
```

Then push `index.html` to GitHub Pages (or any static host).

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # add your API key
python app.py               # starts on http://localhost:5000
```

Open `index.html` directly in your browser. Make sure `API_BASE` is set to `http://localhost:5000`.

## Skin parameters analysed

| Parameter | What it measures |
|---|---|
| Acne & Blemishes | Active breakouts, congestion |
| Hydration | Moisture level, plumpness |
| Redness & Tone | Evenness, flushing |
| Texture & Pores | Surface smoothness, pore size |
| Dark Spots | Pigmentation, sun damage |
