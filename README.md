# mini-kh-OCR API

### Mini-KH-OCR is tool power ai that can extract text from images and pdf that supporting with documents that have subject and reference that using **YOLO11n** for text detection and **CRNN+LSTM+CTC** for text recognition។

---

## Class ID in Text detection

| ID | Class | ខ្មែរ |
|----|-------|--------|
| 0 | subject | កម្មវត្ថុ |
| 1 | reference | យោង |
| 2 | content | អត្ថបទ |

---

## Models

| Role | File |
|------|------|
| Text Detection | `models/khmer-text-detection.pt` |
| Text Recognition | `models/khmerOCR.pt` |

---

## Project Structure

```
mini-kh-OCR/
├── app.py                  ← FastAPI backend
├── requirements.txt        ← Python dependencies
├── .env                    ← model paths & server config
├── .env.example            ← example env file
└── templates/
    └── index.html          ← Web UI
```

---

## Installation

### 1. Clone or download the project

```bash
git clone https://github.com/your-username/mini-kh-OCR.git
cd mini-kh-OCR
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Poppler (for PDF support)

**Windows**
- Download from: https://github.com/oschwartz10612/poppler-windows/releases
- Extract and add the `bin/` folder to your system **PATH**

**Mac**
```bash
brew install poppler
```

**Linux**
```bash
sudo apt install poppler-utils
```

### 4. Setup environment variables

Copy `.env.example` to `.env` and set your model paths:

```bash
cp .env.example .env
```

Edit `.env`:
```
DETECTION_MODEL_PATH=models/khmer-text-detection.pt
OCR_MODEL_PATH=models/khmerOCR.pt

DET_CONF=0.25
DET_IOU=0.45
DET_IMGSZ=640

HOST=127.0.0.1
PORT=8000
```

### 5. Place your model files

```
mini-kh-OCR/
└── models/
    ├── khmer-text-detection.pt
    └── khmerOCR.pt
```

---

## Run the Server

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

You should see:
```
[mini-kh-OCR] Device: cpu
[mini-kh-OCR] Loading detection model from: models/khmer-text-detection.pt
[mini-kh-OCR] Loading OCR model from: models/khmerOCR.pt
[mini-kh-OCR] Both models loaded
INFO:     Uvicorn running on http://127.0.0.1:8000
```

---

## Usage

### Web UI

Open your browser and go to:
```
http://localhost:8000
```

- Drag & drop or select an image (JPG, PNG) or PDF
- Click **ដំណើរការ OCR**
- Results are grouped by class: **កម្មវត្ថុ**, **យោង**, **អត្ថបទ**

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Check server & models status |
| `POST` | `/process` | Upload image or PDF for OCR |
| `GET` | `/docs` | Interactive API docs (Swagger) |

### API Example

```bash
curl -X POST http://localhost:8000/process \
  -F "file=@document.jpg"
```

### Response Format

**Image:**
```json
{
  "type": "image",
  "subject":   ["កម្មវត្ថុ: សំណើរសុំច្បាប់"],
  "reference": ["យោង: លេខ ០០១/២៤"],
  "content":   ["អត្ថបទនៃឯកសារ..."],
  "regions": [
    {
      "id": 0,
      "class_en": "subject",
      "class_km": "កម្មវត្ថុ",
      "conf": 0.91,
      "box": { "x1": 10, "y1": 5, "x2": 320, "y2": 40 },
      "text": "កម្មវត្ថុ: សំណើរសុំច្បាប់"
    }
  ]
}
```

**PDF:**
```json
{
  "type": "pdf",
  "pages": 3,
  "results": [
    {
      "page": 1,
      "subject":   ["..."],
      "reference": ["..."],
      "content":   ["..."],
      "regions":   [...]
    }
  ]
}
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Attribute "main" not found` | Run `uvicorn app:app` not `uvicorn main:app` |
| `FileNotFoundError: model .pt` | Check paths in `.env` are correct |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `pdf2image` error | Install Poppler and add to PATH |
| Port already in use | Change `PORT=8080` in `.env` |

---

## Requirements

- Python 3.8+
- PyTorch
- Ultralytics (YOLO11)
- FastAPI + Uvicorn
- Pillow
- pdf2image + Poppler

---