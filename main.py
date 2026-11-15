"""AquaLens Studio: local water-meter OCR with Qwen2-VL and OpenCV.

The original API contract (/api/analyze and /api/save-results) is preserved.
Models are only initialized when an image is analyzed. No photographs are stored.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
RESULTS_DIR = Path(os.getenv('RESULTS_DIR', str(BASE_DIR / 'results'))).resolve()
MODEL_ID = os.getenv('MODEL_ID', 'Qwen/Qwen2-VL-2B-Instruct')
# First run: set MODEL_LOCAL_ONLY=0 to fetch from Hugging Face. After caching, 1 works offline.
MODEL_LOCAL_ONLY = os.getenv('MODEL_LOCAL_ONLY', '1').strip().lower() not in ('0', 'false', 'no')
MAX_UPLOAD_MB = max(1, int(os.getenv('MAX_UPLOAD_MB', '20')))
MAX_SAVE_RECORDS = 2000
logger = logging.getLogger('aqualens')
app = FastAPI(title='AquaLens Studio - Sayaç Okuma', version='2.0.0')

_model = None
_processor = None
_torch = None
_process_vision_info = None
_load_lock = threading.Lock()
_infer_lock = threading.Lock()
_model_error: str | None = None


def _get_model():
    global _model, _processor, _torch, _process_vision_info, _model_error
    if _model is not None:
        return _model, _processor, _torch, _process_vision_info
    with _load_lock:
        if _model is not None:
            return _model, _processor, _torch, _process_vision_info
        try:
            if MODEL_LOCAL_ONLY:
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            kwargs = {'torch_dtype': dtype, 'local_files_only': MODEL_LOCAL_ONLY}
            if torch.cuda.is_available():
                kwargs['device_map'] = 'auto'
            model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
            processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=MODEL_LOCAL_ONLY)
            model.eval()
            _model, _processor, _torch, _process_vision_info = model, processor, torch, process_vision_info
            _model_error = None
            logger.info('Model ready (%s)', 'offline' if MODEL_LOCAL_ONLY else 'online-capable')
            return _model, _processor, _torch, _process_vision_info
        except Exception as exc:
            # Never place local paths, machine names or HF auth details in client-facing error text.
            logger.warning('Model initialization failed: %s', type(exc).__name__)
            _model_error = ('DEPENDENCIES_MISSING' if isinstance(exc, ImportError) else
                            'MODEL_NOT_CACHED' if MODEL_LOCAL_ONLY else 'MODEL_LOAD_FAILED')
            raise RuntimeError(_model_error) from exc


def get_unrolled_ring(image_bytes: bytes):
    """OpenCV Hough circle + warpPolar ring flattening (same technique as original)."""
    import cv2
    import numpy as np
    from PIL import Image
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('INVALID_IMAGE')
    height, width = img.shape[:2]
    if min(width, height) < 32:
        raise ValueError('IMAGE_TOO_SMALL')
    cx, cy = width // 2, height // 2
    radius = int(min(width, height) * .4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT,
                               1, max(1, height // 4), param1=50, param2=35,
                               minRadius=int(min(width, height) * .25),
                               maxRadius=int(min(width, height) * .5))
    if circles is not None:
        candidate = circles[0, 0]
        cx, cy, radius = int(candidate[0]), int(candidate[1]), int(candidate[2])
    min_r = int(radius * .8)
    max_r = int(radius * 1.25)
    if max_r <= min_r:
        raise ValueError('INVALID_RING')
    unrolled = cv2.warpPolar(img, (max_r, 1600), (cx, cy), max_r, cv2.WARP_POLAR_LINEAR)
    strip = cv2.rotate(unrolled[:, min_r:max_r], cv2.ROTATE_90_CLOCKWISE)
    return Image.fromarray(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB))


def analyze_image(contents: bytes) -> dict:
    from PIL import Image, ImageOps, UnidentifiedImageError
    if not contents:
        raise ValueError('EMPTY_IMAGE')
    try:
        with Image.open(io.BytesIO(contents)) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((1024, 1024))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError('INVALID_IMAGE') from exc
    # Preserve the original image + polar-unwrapped brass-ring dual-image prompt.
    ring_image = get_unrolled_ring(contents)
    model, processor, torch, process_vision_info = _get_model()
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': 'Perform OCR. Read the 5-digit number on the mechanical counter. Return strictly valid JSON.'},
        {'type': 'image', 'image': ring_image},
        {'type': 'text', 'text': 'This is a flattened strip of the outer water-meter ring. Read its 8-digit engraved serial number.'},
        {'type': 'text', 'text': 'Return ONLY JSON: {"m3_degeri":"...","seri_no":"..."}. No other text.'},
    ]}]
    with _infer_lock:  # serialize requests: safe for VRAM-limited local GPUs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos,
                           padding=True, return_tensors='pt').to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=128)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        output = processor.batch_decode(trimmed, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
        del inputs, generated
    match = re.search(r'\{[\s\S]*?\}', output)
    if not match:
        raise ValueError('MODEL_INVALID_JSON')
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError('MODEL_INVALID_JSON') from exc
    if not isinstance(data, dict):
        raise ValueError('MODEL_INVALID_JSON')
    raw_serial = str(data.get('seri_no', 'Okunamadı'))
    serial_match = re.search(r'\b\d{7,8}\b', raw_serial)
    # No municipality/personal label is attached. Only the meter serial is returned.
    return {
        'm3_degeri': str(data.get('m3_degeri', 'Okunamadı'))[:64],
        'seri_no': serial_match.group() if serial_match else raw_serial[:80],
    }


class ResultItem(BaseModel):
    file_name: str = Field(max_length=250)
    status: str = Field(max_length=30)
    m3_degeri: str = Field(max_length=80)
    seri_no: str = Field(max_length=100)
    message: Optional[str] = Field(default=None, max_length=300)


class SaveResultsRequest(BaseModel):
    results: List[ResultItem]
    anonymize: bool = True


def _mask_serial(value: str) -> str:
    value = re.sub(r'\s+', ' ', value or '').strip()
    if len(value) <= 4:
        return '••••' if value else '—'
    return '••••' + value[-4:]


def _row_to_dict(item: ResultItem) -> dict:
    return item.model_dump() if hasattr(item, 'model_dump') else item.dict()


@app.get('/api/status')
def status():
    return {'status': 'ok', 'version': '2.0.0', 'model': MODEL_ID,
            'model_loaded': _model is not None, 'local_only': MODEL_LOCAL_ONLY,
            'model_error': _model_error, 'max_upload_mb': MAX_UPLOAD_MB}


@app.post('/api/analyze')
async def analyze_meter(file: UploadFile = File(...)):
    if file.content_type and not (file.content_type.startswith('image/') or file.content_type == 'application/octet-stream'):
        return JSONResponse({'status': 'error', 'message': 'Yalnızca görsel yükleyebilirsiniz.'}, status_code=415)
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    await file.close()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return JSONResponse({'status': 'error', 'message': f'Dosya {MAX_UPLOAD_MB} MB sınırını aşıyor.'}, status_code=413)
    try:
        result = await run_in_threadpool(analyze_image, data)
        return {'status': 'success', **result}
    except ValueError as exc:
        messages = {
            'INVALID_IMAGE': 'Görsel açılamadı. JPG veya PNG deneyin.',
            'EMPTY_IMAGE': 'Görsel boş.', 'IMAGE_TOO_SMALL': 'Görsel çözünürlüğü çok düşük.',
            'INVALID_RING': 'Sayaç halkası işlenemedi.',
            'MODEL_INVALID_JSON': 'Model beklenen biçimde sonuç üretemedi; yeniden deneyin.',
        }
        return JSONResponse({'status': 'error', 'message': messages.get(str(exc), 'Görsel analiz edilemedi.')}, status_code=422)
    except RuntimeError as exc:
        message = ('Model önbellekte bulunamadı. İlk kurulumda internet bağlıyken '
                   'MODEL_LOCAL_ONLY=0 ile bir kez çalıştırın.' if str(exc) == 'MODEL_NOT_CACHED'
                   else 'Python bağımlılıkları eksik. requirements.txt paketlerini kurun.' if str(exc) == 'DEPENDENCIES_MISSING'
                   else 'Model yüklenemedi. Model kurulumunu ve bağımlılıkları kontrol edin.')
        return JSONResponse({'status': 'error', 'message': message}, status_code=503)
    except Exception:
        logger.exception('Unexpected analysis failure')
        return JSONResponse({'status': 'error', 'message': 'İşlem tamamlanamadı. Sunucu günlüklerini inceleyin.'}, status_code=500)


@app.post('/api/save-results')
async def save_results(payload: SaveResultsRequest):
    if not payload.results:
        raise HTTPException(status_code=400, detail='Kaydedilecek sonuç bulunamadı.')
    if len(payload.results) > MAX_SAVE_RECORDS:
        raise HTTPException(status_code=413, detail='Tek seferde çok fazla kayıt var.')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    name = f'oturum_{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}.json'
    rows = []
    for idx, item in enumerate(payload.results, 1):
        row = _row_to_dict(item)
        if payload.anonymize:
            row['file_name'] = f'goruntu_{idx:03d}'
            row['seri_no'] = _mask_serial(row['seri_no'])
            row['message'] = None  # model errors may contain sensitive input text
        rows.append(row)
    document = {'created_at': now.isoformat(timespec='seconds'),
                'total_count': len(rows),
                'successful_count': sum(x['status'] == 'success' for x in rows),
                'anonymized': payload.anonymize, 'results': rows}
    path = RESULTS_DIR / name
    tmp = RESULTS_DIR / ('.' + name + '.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return {'status': 'success', 'file_name': name,
            'relative_path': str(Path(RESULTS_DIR.name) / name)}


@app.get('/api/history')
def history():
    if not RESULTS_DIR.is_dir():
        return {'status': 'success', 'sessions': []}
    sessions = []
    for path in sorted(RESULTS_DIR.glob('oturum_*.json'), key=lambda p: p.name, reverse=True)[:50]:
        try:
            with path.open(encoding='utf-8') as f:
                entry = json.load(f)
            sessions.append({'id': path.name, 'created_at': entry.get('created_at'),
                             'total_count': entry.get('total_count'),
                             'successful_count': entry.get('successful_count'),
                             'anonymized': entry.get('anonymized', False)})
        except (ValueError, OSError):
            continue
    return {'status': 'success', 'sessions': sessions}


@app.get('/api/history/{session_id}')
def history_detail(session_id: str):
    if not re.fullmatch(r'oturum_[\w-]+\.json', session_id):
        raise HTTPException(status_code=400, detail='Geçersiz oturum kimliği.')
    path = RESULTS_DIR / session_id
    if not path.is_file():
        raise HTTPException(status_code=404, detail='Oturum bulunamadı.')
    try:
        with path.open(encoding='utf-8') as f:
            return json.load(f)
    except (ValueError, OSError):
        raise HTTPException(status_code=500, detail='Oturum okunamadı.')


if STATIC_DIR.exists():
    app.mount('/', StaticFiles(directory=str(STATIC_DIR), html=True), name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8001, reload=False)
