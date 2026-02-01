# Doc Cropper — Deploy to Render (Free Tier)

## What you need
- A [GitHub](https://github.com) account
- A [Render](https://render.com) account (free, sign up with GitHub)

---

## Step 1 — Push to GitHub

Create a new GitHub repo and push these 3 files into the root:

```
your-repo/
├── app.py
├── requirements.txt
└── render.yaml
```

```bash
git init
git add .
git commit -m "initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## Step 2 — Deploy on Render

1. Go to [render.com](https://render.com) → sign in with GitHub
2. Click **New** → **Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` and fills everything in
 (If asked for Start Command : gunicorn app:app --workers 2 --timeout 120 and Build Command : pip install --upgrade pip && pip install -r requirements.txt)  
5. Click **Deploy**

Wait 2–3 minutes for the first build. Render installs dependencies and starts gunicorn.

Your public URL will be:
```
https://YOUR-SERVICE-NAME.onrender.com
```

---

## Step 3 — Test with curl

### Crop a document image

```bash
curl -X POST https://YOUR-SERVICE-NAME.onrender.com/crop \
  -F "file=@/path/to/your/document.jpg"
```

Response:
```json
{
  "success": true,
  "corners_detected": [[295, 50], [891, 50], [295, 849], [899, 849]],
  "cropped_color_image_url": "/outputs/uuid_cropped_color.png",
  "cropped_bw_image_url":    "/outputs/uuid_cropped_bw.png",
  "pdf_url":                 "/outputs/uuid_cropped.pdf"
}
```

### Download the outputs

Replace `YOUR-SERVICE-NAME` and the filenames from the response above:

```bash
# Colour version
curl -o color.png https://YOUR-SERVICE-NAME.onrender.com/outputs/uuid_cropped_color.png

# Black & white (clean scan look)
curl -o bw.png https://YOUR-SERVICE-NAME.onrender.com/outputs/uuid_cropped_bw.png

# PDF
curl -o document.pdf https://YOUR-SERVICE-NAME.onrender.com/outputs/uuid_cropped.pdf
```

### Health check (confirm the service is running)

```bash
curl https://YOUR-SERVICE-NAME.onrender.com/health
# {"status": "ok"}
```

---

## Supported file types

`png`, `jpg`, `jpeg`, `webp`, `bmp`, `tiff`, `pdf`

PDF uploads are converted to an image automatically (first page only).

---

## Free tier limits (Render)

| Limit | Value |
|---|---|
| Requests per month | Unlimited |
| Compute | 750 CPU-seconds/month shared across free services |
| Sleep | Service sleeps after 15 min of inactivity; cold start adds ~3s to next request |
| Storage | Ephemeral — files are deleted when the container restarts |

The sleep behaviour is the main thing to know. If nobody has hit the API in 15 minutes, the next request will be ~3 seconds slower while the container wakes up. After that it's normal speed.

Because storage is ephemeral, output files only exist until the next deploy or restart. Download them right after the `/crop` call.

---

## File layout

| File | Purpose |
|---|---|
| `app.py` | Flask app — the full crop pipeline |
| `requirements.txt` | Python dependencies |
| `render.yaml` | Render service config (plan, build cmd, start cmd) |
