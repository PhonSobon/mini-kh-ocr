"""
mini-kh-OCR API
FastAPI backend — text detection (YOLO11n) + text recognition (CRNN+CTC)
Models loaded from environment variables.
"""

import os
import io
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()


DETECTION_MODEL_PATH = os.getenv("DETECTION_MODEL_PATH")
OCR_MODEL_PATH       = os.getenv("OCR_MODEL_PATH")

CLASS_NAMES = {
    0: {"en": "subject",   "km": "កម្មវត្ថុ"},
    1: {"en": "reference", "km": "យោង"},
    2: {"en": "content",   "km": "អត្ថបទ"},
}

TOKENS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឩឪឫឬឭឮឯឰឱឲឳ"
    "ាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៑្។៕៖ៗ៘៛៝"
    "០១២៣៤៥៦៧៨៩៳"
    "!@#$%^&*()-_=+[]{};:'\",.<>?/|\\ "
)
NUM_CHARS = len(TOKENS)
IDX2CHAR  = {i + 1: c for i, c in enumerate(TOKENS)}


class KhmerOCR(nn.Module):
    def __init__(self, num_chars=NUM_CHARS, hidden_size=256):
        super().__init__()
        self.cnn = nn.Sequential(
            self._conv(1, 32),  nn.MaxPool2d(2, 2),
            self._conv(32, 64), nn.MaxPool2d(2, 2),
            self._conv(64, 128),
            self._conv(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),
            self._conv(128, 256),
            self._conv(256, 256),
            nn.MaxPool2d((4, 1), (4, 1)),
        )
        self.lstm1 = nn.LSTM(256, hidden_size, bidirectional=True, batch_first=True)
        self.fc1   = nn.Linear(hidden_size * 2, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc    = nn.Linear(hidden_size * 2, num_chars + 1)

    def _conv(self, i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1, bias=False),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm2(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[mini-kh-OCR] Device: {device}")

print(f"[mini-kh-OCR] Loading detection model from: {DETECTION_MODEL_PATH}")
detector = YOLO(DETECTION_MODEL_PATH)

print(f"[mini-kh-OCR] Loading OCR model from: {OCR_MODEL_PATH}")
ocr_model = KhmerOCR(NUM_CHARS).to(device)
ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=device))
ocr_model.eval()

print("[mini-kh-OCR] Both models loaded")


def preprocess_crop(crop: Image.Image) -> torch.Tensor:
    img = crop.convert("L")
    w, h = img.size
    new_w = max(1, int(w / h * 32))
    img = img.resize((new_w, 32))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(device)


def ctc_decode(logits: torch.Tensor) -> str:
    preds = torch.argmax(logits, dim=2)[:, 0].cpu().numpy()
    prev, text = -1, []
    for p in preds:
        if p != prev and p != 0:
            text.append(IDX2CHAR.get(int(p), ""))
        prev = p
    return "".join(text)


def process_image(pil_img: Image.Image) -> dict:
    """Run detection + OCR on one PIL image. Returns structured result."""
    results = detector.predict(
        source=pil_img,
        conf=float(os.getenv("DET_CONF", "0.25")),
        iou=float(os.getenv("DET_IOU",  "0.45")),
        imgsz=int(os.getenv("DET_IMGSZ", "640")),
        verbose=False,
    )

    boxes   = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
    cls_ids = [int(c) for c in results[0].boxes.cls.cpu().numpy()]
    confs   = [float(c) for c in results[0].boxes.conf.cpu().numpy()]

    # Sort top → bottom
    order   = sorted(range(len(boxes)), key=lambda i: boxes[i][1])
    boxes   = [boxes[i]   for i in order]
    cls_ids = [cls_ids[i] for i in order]
    confs   = [confs[i]   for i in order]

    output = {
        "subject":   [],   # កម្មវត្ថុ
        "reference": [],   # យោង
        "content":   [],   # អត្ថបទ
        "regions":   [],
    }

    for box, cls_id, conf in zip(boxes, cls_ids, confs):
        x1, y1, x2, y2 = box
        crop  = pil_img.crop((x1, y1, x2, y2))
        tensor = preprocess_crop(crop)

        with torch.no_grad():
            logits = ocr_model(tensor)
        text = ctc_decode(logits)

        cls_info = CLASS_NAMES.get(cls_id, {"en": "unknown", "km": ""})
        key = cls_info["en"]

        if key in output:
            output[key].append(text)

        output["regions"].append({
            "id":        cls_id,
            "class_en":  cls_info["en"],
            "class_km":  cls_info["km"],
            "conf":      round(conf, 3),
            "box":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "text":      text,
        })

    return output


def pdf_to_images(data: bytes) -> List[Image.Image]:
    """Convert PDF bytes → list of PIL images (one per page)."""
    try:
        from pdf2image import convert_from_bytes
        return convert_from_bytes(data, dpi=200)
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pdf2image not installed. Run: pip install pdf2image"
        )

app = FastAPI(
    title="mini-kh-OCR API",
    description="Khmer document OCR — text detection + recognition",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def ui():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(device),
        "detection_model": DETECTION_MODEL_PATH,
        "ocr_model": OCR_MODEL_PATH,
    }


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """
    Upload an image (JPG/PNG) or PDF.
    Returns OCR results grouped by detection class.
    """
    data     = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".pdf"):
            pages = pdf_to_images(data)
            all_pages = []
            for page_num, page_img in enumerate(pages, 1):
                page_result = process_image(page_img)
                all_pages.append({
                    "page":      page_num,
                    "subject":   page_result["subject"],
                    "reference": page_result["reference"],
                    "content":   page_result["content"],
                    "regions":   page_result["regions"],
                })
            return JSONResponse({
                "type":    "pdf",
                "pages":   len(pages),
                "results": all_pages,
            })

        else:
            pil_img = Image.open(io.BytesIO(data)).convert("RGB")
            result  = process_image(pil_img)
            return JSONResponse({
                "type":      "image",
                "subject":   result["subject"],
                "reference": result["reference"],
                "content":   result["content"],
                "regions":   result["regions"],
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )