from __future__ import annotations

import os
import io
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, send_file
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chinese ID cards are 85.60 mm × 54.00 mm. Render at 300 DPI for a crisp PDF.
CARD_WIDTH_MM = 85.6
CARD_HEIGHT_MM = 54.0
CARD_WIDTH_PX = 1011
CARD_HEIGHT_PX = 638
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 2


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _candidate_quad(
    contour: np.ndarray,
    image_area: float,
    image_width: int,
    image_height: int,
) -> tuple[float, np.ndarray] | None:
    perimeter = cv2.arcLength(contour, True)
    # Rounded ID-card corners often need a little more simplification than a
    # sharp paper rectangle. Try a slightly wider tolerance before giving up.
    approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    area = abs(cv2.contourArea(approx))
    if area < image_area * 0.08 or area > image_area * 0.92:
        return None
    points = approx.reshape(4, 2).astype("float32")
    # A contour coinciding with the photograph boundary is the whole photo,
    # not the card. Never use that as a crop fallback.
    margin_x = max(2.0, image_width * 0.008)
    margin_y = max(2.0, image_height * 0.008)
    if (
        np.any(points[:, 0] <= margin_x)
        or np.any(points[:, 0] >= image_width - margin_x)
        or np.any(points[:, 1] <= margin_y)
        or np.any(points[:, 1] >= image_height - margin_y)
    ):
        return None
    return area, points


def _score_candidate(area: float, points: np.ndarray, image_area: float, target_ratio: float) -> float:
    rect = cv2.minAreaRect(points)
    rw, rh = rect[1]
    if min(rw, rh) <= 0:
        return 0.0
    ratio = max(rw, rh) / min(rw, rh)
    ratio_penalty = abs(np.log(max(ratio, 1e-6) / target_ratio))
    return (area / image_area) / (1.0 + ratio_penalty * 1.8)


def _find_candidates(mask: np.ndarray, image_area: float, image_width: int, image_height: int) -> list[tuple[float, np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_ratio = CARD_WIDTH_MM / CARD_HEIGHT_MM
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
        candidate = _candidate_quad(contour, image_area, image_width, image_height)
        if not candidate:
            continue
        area, points = candidate
        score = _score_candidate(area, points, image_area, target_ratio)
        if score > 0:
            candidates.append((score, points))
    return candidates


def _detect_card(image: np.ndarray) -> np.ndarray:
    """Return four card corners in the original image coordinate system."""
    original_h, original_w = image.shape[:2]
    scale = min(1.0, 1600.0 / max(original_h, original_w))
    working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 140)
    image_area = float(working.shape[0] * working.shape[1])
    working_h, working_w = working.shape[:2]
    candidates: list[tuple[float, np.ndarray]] = []
    edge_kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, edge_kernel, iterations=1)
    candidates.extend(_find_candidates(edges, image_area, working_w, working_h))

    # A white ID card on a dark desk often has no reliable Canny outline:
    # the security pattern and wood grain produce more edges than the border.
    # Bright-region masks make the card itself the dominant contour.
    for threshold in (130, 150, 170, 190):
        bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
        candidates.extend(_find_candidates(bright, image_area, working_w, working_h))

    # Otsu is useful when exposure differs greatly between photos.
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    candidates.extend(_find_candidates(bright, image_area, working_w, working_h))

    if candidates:
        points = max(candidates, key=lambda item: item[0])[1]
    else:
        raise ValueError("没有找到足够大的证件轮廓，请让证件完整出现在画面内。")

    if scale < 1:
        points /= scale
    return _order_points(points)


def crop_card(image: np.ndarray) -> np.ndarray:
    corners = _detect_card(image)
    destination = np.array(
        [
            [0, 0],
            [CARD_WIDTH_PX - 1, 0],
            [CARD_WIDTH_PX - 1, CARD_HEIGHT_PX - 1],
            [0, CARD_HEIGHT_PX - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (CARD_WIDTH_PX, CARD_HEIGHT_PX), borderMode=cv2.BORDER_REPLICATE)


def _read_upload(upload) -> np.ndarray:
    raw = upload.read()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("图片为空或超过 25MB，请重新选择。")
    try:
        # OpenCV ignores EXIF orientation. Phone photos commonly store the
        # camera rotation in EXIF, so transpose before handing pixels to CV.
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        image = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
    except (OSError, ValueError):
        raise ValueError("无法读取图片，请使用 JPG、PNG 或 HEIC 转换后的图片。")
    return image


def create_pdf(front: np.ndarray, back: np.ndarray) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = OUTPUT_DIR / f"身份证-正反面-{timestamp}-{uuid.uuid4().hex[:6]}.pdf"
    with tempfile.TemporaryDirectory(prefix="qiuqiu-id-card-") as temp_dir:
        front_path = Path(temp_dir) / "front.jpg"
        back_path = Path(temp_dir) / "back.jpg"
        cv2.imwrite(str(front_path), front, [cv2.IMWRITE_JPEG_QUALITY, 96])
        cv2.imwrite(str(back_path), back, [cv2.IMWRITE_JPEG_QUALITY, 96])

        page_width, page_height = A4
        card_width = CARD_WIDTH_MM * mm
        card_height = CARD_HEIGHT_MM * mm
        margin = 20 * mm
        gap = 18 * mm
        x = (page_width - card_width) / 2
        top_y = page_height - margin - card_height
        bottom_y = top_y - gap - card_height

        pdf = canvas.Canvas(str(output_path), pagesize=A4)
        pdf.drawImage(ImageReader(str(front_path)), x, top_y, width=card_width, height=card_height, mask="auto")
        pdf.drawImage(ImageReader(str(back_path)), x, bottom_y, width=card_width, height=card_height, mask="auto")
        pdf.showPage()
        pdf.save()
    return output_path


@app.get("/")
def index():
    return render_template("index.html", error=None, result=None)


@app.post("/process")
def process():
    front_upload = request.files.get("front")
    back_upload = request.files.get("back")
    if not front_upload or not back_upload:
        return render_template("index.html", error="请同时上传身份证正面和反面照片。", result=None), 400

    try:
        front = crop_card(_read_upload(front_upload))
        back = crop_card(_read_upload(back_upload))
        output_path = create_pdf(front, back)
    except (ValueError, cv2.error) as exc:
        return render_template("index.html", error=str(exc), result=None), 400

    return render_template("index.html", error=None, result={"name": output_path.name, "url": f"/download/{output_path.name}"})


@app.get("/download/<path:filename>")
def download(filename: str):
    safe_path = (OUTPUT_DIR / filename).resolve()
    if safe_path.parent != OUTPUT_DIR.resolve() or not safe_path.is_file():
        return "文件不存在", 404
    return send_file(safe_path, as_attachment=True, download_name=safe_path.name, mimetype="application/pdf")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8765")), debug=False)
