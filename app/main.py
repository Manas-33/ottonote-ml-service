from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="OttoNote ML Service", version="0.1.0")

# CORS - adjust in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionResponse(BaseModel):
    text: str
    language: Optional[str] = None
    duration_sec: Optional[float] = None
    model: Optional[str] = None

class TranscriptionRequest(BaseModel):
    file_path: str


_whisper_model = None
_whisper_model_name = os.getenv("WHISPER_MODEL", "base")


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import whisper. Ensure 'openai-whisper' is installed."
            ) from exc
        _whisper_model = whisper.load_model(_whisper_model_name)
    return _whisper_model

# For local mp3 files
def transcribe_file_path(file_path: str) -> TranscriptionResponse:
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    valid_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac'}
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in valid_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: {file_ext}")
    
    try:
        model = get_whisper_model()
        result = model.transcribe(file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}") from exc

    text = result.get("text", "").strip()
    language = result.get("language")
    duration = None
    try:
        segments = result.get("segments") or []
        if segments:
            duration = float(segments[-1].get("end") or 0.0)
    except Exception:
        duration = None

    return TranscriptionResponse(
        text=text,
        language=language,
        duration_sec=duration,
        model=_whisper_model_name,
    )


@app.get("/health")
def health():
    return {"status": "ok"}

# For HTTP file requests
@app.post("/v1/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe")
):
    if not file.content_type or not any(
        x in file.content_type for x in ("audio", "video", "octet-stream")
    ):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        suffix = os.path.splitext(file.filename or "upload")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file_bytes = await file.read()
            tmp.write(file_bytes)
            tmp_path = tmp.name

        model = get_whisper_model()
        result = model.transcribe(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    text = result.get("text", "").strip()
    language = result.get("language")
    duration = None
    try:
        segments = result.get("segments") or []
        if segments:
            duration = float(segments[-1].get("end") or 0.0)
    except Exception:
        duration = None

    return TranscriptionResponse(
        text=text,
        language=language,
        duration_sec=duration,
        model=_whisper_model_name,
    )

@app.post("/v1/transcribe-path", response_model=TranscriptionResponse)
async def transcribe_from_path(request: TranscriptionRequest):
    """Transcribe audio from a local file path"""
    return transcribe_file_path(request.file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)