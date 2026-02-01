import os
import uuid
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from scipy.spatial import distance


# ─────────────────────────────────────────────
# App Factory & Config
# ─────────────────────────────────────────────
def create_app():
    app = Flask(__name__, static_folder='uploads', static_url_path='/uploads')
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(__file__), 'outputs')
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'pdf'}

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    return app

app = create_app()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_id():
    return str(uuid.uuid4())


# ─────────────────────────────────────────────
# STEP 1 — Pre-processing
# ─────────────────────────────────────────────
# Goal: produce a clean binary edge map so corner detection
# can reliably find the document quadrilateral.
#
# Pipeline:
#   Grayscale → Denoise → Morphological Close → Canny → Dilate → Erode
#
# Why each step matters:
#   - fastNlMeansDenoising: removes sensor noise without destroying edges
#     (better than Gaussian blur for photos of documents)
#   - Morphological close (ellipse kernel, 5 iterations): fills small gaps
#     in the document border caused by shadows or creases
#   - Canny: produces thin, one-pixel-wide edges — ideal for contour finding
#   - Dilate then Erode: thickens broken edge segments so they connect,
#     making findContours pick up the full document outline
# ─────────────────────────────────────────────
def preprocess(img):
    # 1 — Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2 — Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 3 — Morphological close: seal gaps in the document border
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=5)

    # 4 — Canny edge detection
    edges = cv2.Canny(morphed, 75, 200)

    # 5 — Dilate then erode to connect broken edge segments
    kernel = np.ones((5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    result  = cv2.erode(dilated, kernel, iterations=1)

    return result


# ─────────────────────────────────────────────
# STEP 2 — Corner Detection
# ─────────────────────────────────────────────
# Finds the 4 corners of the largest quadrilateral contour.
#
# Key decisions:
#   - RETR_EXTERNAL: only outermost contours (ignores text, logos, etc.)
#   - Area threshold (6000): filters out small noise contours
#   - approxPolyDP with epsilon = 0.05 * perimeter: simplifies the contour
#     into a polygon; we only keep it if it simplifies to exactly 4 points
#   - We track maxArea so we always get the BIGGEST quad — that's the paper
# ─────────────────────────────────────────────
def detect_corners(img_processed):
    corners = np.array([])
    max_area = 0

    contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 6000:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Only accept quadrilaterals, and only if bigger than current best
            if len(approx) == 4 and area > max_area:
                corners = approx
                max_area = area

    if len(corners) == 0:
        return None
    return corners


# ─────────────────────────────────────────────
# STEP 2b — Sort Corners
# ─────────────────────────────────────────────
# Reorders the 4 detected points into the exact order the warper needs:
#   [0] top-left      → smallest (x + y)
#   [1] top-right     → smallest (y - x)  i.e. min of diff
#   [2] bottom-left   → largest  (y - x)  i.e. max of diff
#   [3] bottom-right  → largest  (x + y)
#
# This specific ordering matches the destination rectangle in the warper:
#   [[0,0], [W,0], [0,H], [W,H]]
# ─────────────────────────────────────────────
def sort_corners(corners):
    corners = corners.reshape((4, 2))
    sorted_corners = np.zeros((4, 1, 2), np.int32)

    # Sum of coordinates: top-left is smallest, bottom-right is largest
    add = corners.sum(axis=1)
    sorted_corners[0] = corners[np.argmin(add)]   # top-left
    sorted_corners[3] = corners[np.argmax(add)]   # bottom-right

    # Difference (y - x): top-right is most negative, bottom-left is most positive
    diff = np.diff(corners, axis=1)
    sorted_corners[1] = corners[np.argmin(diff)]  # top-right
    sorted_corners[2] = corners[np.argmax(diff)]  # bottom-left

    return sorted_corners


# ─────────────────────────────────────────────
# STEP 3 — Perspective Warp
# ─────────────────────────────────────────────
# Transforms the skewed document into a flat rectangle.
#
# How it works:
#   - Source points: the 4 detected corners on the original image
#   - Destination points: a clean rectangle [0,0] → [W,H]
#   - getPerspectiveTransform computes the 3×3 matrix that maps
#     source → destination
#   - warpPerspective applies that matrix to every pixel
#
# The output width/height are passed in (caller decides resolution).
# ─────────────────────────────────────────────
def warp(img, corners, width, height):
    corners = sort_corners(corners)

    src = np.float32(corners)
    dst = np.float32([
        [0, 0],
        [width, 0],
        [0, height],
        [width, height]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (width, height))

    return warped


# ─────────────────────────────────────────────
# STEP 4 — Post-processing
# ─────────────────────────────────────────────
# Makes the warped document look like a clean scan.
#
# Pipeline:
#   Grayscale → Denoise (h=15, stronger than pre-processing)
#              → Adaptive Threshold
#
# Why adaptive threshold instead of global (Otsu)?
#   Photos of documents have uneven lighting — the top may be brighter
#   than the bottom. Adaptive threshold computes a local threshold for
#   each pixel based on its neighbourhood, so it handles this gracefully.
#
# Result: a crisp black-on-white binary image that looks like a photocopy.
# ─────────────────────────────────────────────
def postprocess(img_warped):
    # Grayscale
    gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

    # Stronger denoising pass on the warped image
    # h=10 here instead of 15 — we want to preserve fine detail in the text,
    # not over-smooth it before thresholding
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Sharpen slightly so thin strokes stay crisp after denoising
    kernel_sharp = np.array([[ 0, -1,  0],
                             [-1,  5, -1],
                             [ 0, -1,  0]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, kernel_sharp)

    # Adaptive threshold → clean black & white
    #
    # blockSize=21: the neighbourhood window used to compute the local mean.
    #   At 11 it was too small for our 2× upscaled image — each pixel's
    #   threshold was being set by only ~100 neighbours, so shadow gradients
    #   across the page were bleeding into the text. 21 gives a much more
    #   stable local reference.
    #
    # C=10: how much to subtract from the local mean before thresholding.
    #   At C=2 anything even slightly darker than its neighbours became black,
    #   which turned grey ink AND shadow into solid black and made strokes
    #   bloat together. C=10 means only genuinely dark pixels (actual ink)
    #   survive — mid-greys from lighting variation become white.
    result = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    return result


# ─────────────────────────────────────────────
# PDF Generation
# ─────────────────────────────────────────────
def generate_pdf(image_path, pdf_path):
    """Embeds the cropped image into a single-page PDF, scaled to fit."""
    pil_img = Image.open(image_path)
    img_w, img_h = pil_img.size

    page_w, page_h = letter       # 612 × 792 points
    margin = 40
    avail_w = page_w - 2 * margin
    avail_h = page_h - 2 * margin

    # Scale uniformly to fit, preserving aspect ratio
    scale  = min(avail_w / img_w, avail_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale

    # Center on page
    x = margin + (avail_w - draw_w) / 2
    y = margin + (avail_h - draw_h) / 2

    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawImage(image_path, x, y, width=draw_w, height=draw_h)
    c.save()
    return True


# ─────────────────────────────────────────────
# Route: POST /crop
# ─────────────────────────────────────────────
# Full pipeline in one call:
#   Upload → Preprocess → Detect corners → Warp → Postprocess
#   → Save cropped image + PDF → Return both URLs
#
# Response JSON:
#   {
#     "success": true,
#     "corners_detected": [[x,y], ...],   ← the 4 raw detected points
#     "cropped_image_url": "/outputs/...",
#     "cropped_bw_image_url": "/outputs/...",  ← post-processed B&W version
#     "pdf_url": "/outputs/..."
#   }
# ─────────────────────────────────────────────
@app.route('/crop', methods=['POST'])
def crop_document():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File type not allowed."}), 415

    # ── Save upload ──
    file_id = generate_id()
    ext = file.filename.rsplit('.', 1)[1].lower()
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
    file.save(upload_path)

    # ── If PDF, rasterise first page to PNG ──
    if ext == 'pdf':
        from pdf2image import convert_from_path
        try:
            pages = convert_from_path(upload_path, first_page=1, last_page=1, dpi=300)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.png")
            pages[0].save(upload_path, 'PNG')
        except Exception as e:
            return jsonify({"success": False, "error": f"PDF conversion failed: {str(e)}"}), 500

    # ── Read image ──
    img = cv2.imread(upload_path)
    if img is None:
        return jsonify({"success": False, "error": "Could not read image file."}), 500

    # Resize for consistent processing (keeps aspect ratio)
    h, w = img.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # ── STEP 1: Preprocess ──
    processed = preprocess(img)

    # ── STEP 2: Detect corners ──
    corners = detect_corners(processed)
    if corners is None:
        return jsonify({
            "success": False,
            "error": "Could not detect document corners. "
                     "Ensure the document is clearly visible with distinct edges against the background."
        }), 422

    # ── STEP 3: Warp ──
    # Output resolution: average opposite edges so the destination rectangle
    # is a true flat rectangle regardless of camera angle.
    #
    # sorted_pts order: [0]=TL  [1]=TR  [2]=BL  [3]=BR
    #
    # If we only used the top edge for width, and the bottom edge is a
    # different length (paper photographed at an angle), the warp destination
    # wouldn't match — leaving a gap or skew at the bottom.
    # Averaging both pairs fixes that.
    sorted_pts = sort_corners(corners).reshape(4, 2)

    # Width: average of top edge (TL→TR) and bottom edge (BL→BR)
    width_top = np.hypot(sorted_pts[1][0] - sorted_pts[0][0],
                         sorted_pts[1][1] - sorted_pts[0][1])
    width_bot = np.hypot(sorted_pts[3][0] - sorted_pts[2][0],
                         sorted_pts[3][1] - sorted_pts[2][1])
    out_w = int(max(width_top, width_bot))

    # Height: average of left edge (TL→BL) and right edge (TR→BR)
    height_left  = np.hypot(sorted_pts[2][0] - sorted_pts[0][0],
                            sorted_pts[2][1] - sorted_pts[0][1])
    height_right = np.hypot(sorted_pts[3][0] - sorted_pts[1][0],
                            sorted_pts[3][1] - sorted_pts[1][1])
    out_h = int(max(height_left, height_right))

    # Scale up for better quality in the output
    scale_factor = 2
    out_w *= scale_factor
    out_h *= scale_factor

    warped = warp(img, corners, out_w, out_h)

    # ── Save warped (colour) image ──
    cropped_color_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_cropped_color.png")
    cv2.imwrite(cropped_color_path, warped)

    # ── STEP 4: Post-process → clean B&W scan ──
    warped_bw = postprocess(warped)

    cropped_bw_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_cropped_bw.png")
    cv2.imwrite(cropped_bw_path, warped_bw)

    # ── Generate PDF (from the B&W version) ──
    pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_cropped.pdf")
    generate_pdf(cropped_bw_path, pdf_path)

    return jsonify({
        "success": True,
        "corners_detected": corners.reshape(4, 2).tolist(),
        "cropped_color_image_url": f"/outputs/{file_id}_cropped_color.png",
        "cropped_bw_image_url":    f"/outputs/{file_id}_cropped_bw.png",
        "pdf_url":                 f"/outputs/{file_id}_cropped.pdf"
    })


# ─────────────────────────────────────────────
# Route: GET /outputs/<filename>  — serve any output file
# ─────────────────────────────────────────────
@app.route('/outputs/<filename>', methods=['GET'])
def serve_output(filename):
    return send_file(
        os.path.join(app.config['OUTPUT_FOLDER'], filename),
        as_attachment=True
    )


# ─────────────────────────────────────────────
# Route: GET /health  — Render health check
# ─────────────────────────────────────────────
# Render pings this every 30s. Must return 200 or the service gets restarted.
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


# ─────────────────────────────────────────────
# Route: POST /get_similar  (kept from original)
# ─────────────────────────────────────────────
@app.route('/get_similar', methods=['POST'])
def cosine_similarity():
    data = request.json
    query_vector = data['query_vector']
    vector_text_pairs = data['vectors']

    vectors = [pair['embeddings'] for pair in vector_text_pairs]
    texts   = [pair['text'] for pair in vector_text_pairs]

    most_similar_index = max(
        range(len(vectors)),
        key=lambda i: 1 - distance.cosine(query_vector, vectors[i])
    )
    return jsonify({'most_similar_text': texts[most_similar_index]})


# ─────────────────────────────────────────────
# Render assigns a port via the PORT env var.
# Flask MUST bind to that port or Render can't route traffic to the service.
# ─────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
