from __future__ import annotations

import os
import tempfile
from typing import Optional, List
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
# ~/.cache/whisper/


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

# ---------------------- Diarization (pyannote) ----------------------
_diarization_pipeline = None
_hf_token = os.getenv("HUGGINGFACE_TOKEN", "")


class DiarizationTurn(BaseModel):
    start: float
    end: float
    speaker: str


class DiarizationResponse(BaseModel):
    turns: List[DiarizationTurn]
    model: Optional[str] = None


def get_diarization_pipeline():
    global _diarization_pipeline
    if _diarization_pipeline is None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Failed to import pyannote.audio. Ensure it is installed."
            ) from exc

        if not _hf_token:
            raise RuntimeError(
                "Missing HF token. Set HF_TOKEN or HUGGINGFACE_TOKEN env var to access pyannote models."
            )

        # Default to pretrained diarization pipeline
        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=_hf_token,
        )
    return _diarization_pipeline


def diarize_file_path(file_path: str) -> DiarizationResponse:
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    try:
        pipeline = get_diarization_pipeline()
        diarization = pipeline(file_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Diarization failed: {str(exc)}") from exc

    turns: List[DiarizationTurn] = []
    try:
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(
                DiarizationTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(speaker),
                )
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse diarization: {str(exc)}") from exc

    return DiarizationResponse(turns=turns, model="pyannote/speaker-diarization-3.1")


# ---------------- Speaker-attributed transcription (Combine) ----------------
class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str


class SpeakerTranscriptResponse(BaseModel):
    segments: List[SpeakerSegment]
    full_text: str
    asr_model: str
    diarization_model: str


# ---------------- Summarization (Gemini) ----------------
class MeetingSummaryResponse(BaseModel):
    overall_summary: str
    key_points: List[str]
    action_items: List[str]
    decisions: List[str]
    total_duration: float
    speakers: List[str]

def _assign_speakers_to_whisper_segments(
    whisper_segments: List[dict], diarization_turns: List[DiarizationTurn]
) -> List[SpeakerSegment]:
    assigned: List[SpeakerSegment] = []
    for seg in whisper_segments:
        ws = float(seg.get("start", 0.0))
        we = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for turn in diarization_turns:
            ts, te = turn.start, turn.end
            overlap = max(0.0, min(we, te) - max(ws, ts))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker
        assigned.append(
            SpeakerSegment(start=ws, end=we, speaker=best_speaker, text=text)
        )
    return assigned


def transcribe_diarize_file_path(file_path: str) -> SpeakerTranscriptResponse:
    model = get_whisper_model()
    asr_result = model.transcribe(file_path)
    whisper_segments: List[dict] = asr_result.get("segments") or []

    diar_response = diarize_file_path(file_path)

    speaker_segments = _assign_speakers_to_whisper_segments(
        whisper_segments, diar_response.turns
    )
    full_text = (asr_result.get("text") or "").strip()

    return SpeakerTranscriptResponse(
        segments=speaker_segments,
        full_text=full_text,
        asr_model=_whisper_model_name,
        diarization_model="pyannote/speaker-diarization-3.1",
    )

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


@app.post("/v1/diarize", response_model=DiarizationResponse)
async def diarize_audio(file: UploadFile = File(..., description="Audio file to diarize")):
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

        response = diarize_file_path(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return response


class DiarizationRequest(BaseModel):
    file_path: str


@app.post("/v1/diarize-path", response_model=DiarizationResponse)
async def diarize_from_path(request: DiarizationRequest):
    return diarize_file_path(request.file_path)


@app.post("/v1/transcribe-diarize", response_model=SpeakerTranscriptResponse)
async def transcribe_diarize(
    file: UploadFile = File(..., description="Audio file to transcribe and diarize")
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

        response = transcribe_diarize_file_path(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return response

_gemini_client = None
_gemini_api_key = os.getenv("GEMINI_API_KEY")

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError(
                "Failed to import google.genai. Ensure 'google-genai' is installed."
            ) from exc
        
        if not _gemini_api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY environment variable."
            )
        
        _gemini_client = genai.Client(api_key=_gemini_api_key)
    return _gemini_client

def _format_transcript_for_summarization(segments: List[SpeakerSegment]) -> str:
    """Format speaker segments into readable transcript"""
    lines = []
    for segment in segments:
        speaker = segment.speaker
        text = segment.text.strip()
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _summarize_with_gemini(transcript: str) -> MeetingSummaryResponse:
    """Summarize the full transcript using Gemini"""
    try:
        client = get_gemini_client()
        
        prompt = f"""
        Analyze this meeting transcript and respond with ONLY a valid JSON object. Do not include any other text before or after the JSON.

        Required JSON format:
        {{
            "overall_summary": "Brief 2-3 sentence summary of the meeting",
            "key_points": ["key point 1", "key point 2", "key point 3"],
            "action_items": ["action item 1", "action item 2"],
            "decisions": ["decision 1", "decision 2"]
        }}

        Meeting transcript:
        {transcript}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[prompt],
            config={
                "temperature": 0.3,
                "max_output_tokens": 2000,
            }
        )
        import json
        response_text = response.text.strip()
        if not response_text:
            raise ValueError("Empty response from Gemini")
        
        try:
            if response_text.startswith('```json'):
                end_marker = response_text.find('```', 7)
                if end_marker != -1:
                    response_text = response_text[7:end_marker].strip()
            elif response_text.startswith('```'):
                end_marker = response_text.find('```', 3)
                if end_marker != -1:
                    response_text = response_text[3:end_marker].strip()
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_text = response_text[start_idx:end_idx]
            result = json.loads(json_text)
        except json.JSONDecodeError as json_err:
            print(f"JSON parsing failed: {json_err}")
            print(f"Response text: {response_text[:500]}...")
            result = {
                "overall_summary": response_text[:200] + "..." if len(response_text) > 200 else response_text,
                "key_points": [],
                "action_items": [],
                "decisions": []
            }
        
        return MeetingSummaryResponse(
            overall_summary=result.get("overall_summary", ""),
            key_points=result.get("key_points", []),
            action_items=result.get("action_items", []),
            decisions=result.get("decisions", []),
            total_duration=0.0,  
            speakers=[] 
        )
        
    except Exception as exc:
        return MeetingSummaryResponse(
            overall_summary=f"Meeting summary unavailable due to error: {str(exc)}",
            key_points=[],
            action_items=[],
            decisions=[],
            total_duration=0.0,
            speakers=[]
        )

def summarize_transcript(segments: List[SpeakerSegment]) -> MeetingSummaryResponse:
    """Main function to summarize a transcript directly without chunking"""
    if not segments:
        return MeetingSummaryResponse(
            overall_summary="No content to summarize",
            key_points=[],
            action_items=[],
            decisions=[],
            total_duration=0.0,
            speakers=[]
        )
    total_duration = segments[-1].end if segments else 0.0
    transcript_text = _format_transcript_for_summarization(segments)
    summary = _summarize_with_gemini(transcript_text)
    summary.total_duration = total_duration
    summary.speakers = list(set([seg.speaker for seg in segments]))
    return summary


class TranscribeDiarizeRequest(BaseModel):
    file_path: str


@app.post("/v1/transcribe-diarize-path", response_model=SpeakerTranscriptResponse)
async def transcribe_diarize_path(request: TranscribeDiarizeRequest):
    return transcribe_diarize_file_path(request.file_path)


@app.post("/v1/summarize", response_model=MeetingSummaryResponse)
async def summarize_audio(file: UploadFile = File(..., description="Audio file to transcribe, diarize, and summarize")):
    """Complete pipeline: transcribe + diarize + summarize"""
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
        transcript_response = transcribe_diarize_file_path(tmp_path)
        summary_response = summarize_transcript(transcript_response.segments)
        
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    return summary_response
class SummarizeRequest(BaseModel):
    file_path: str

@app.post("/v1/summarize-path", response_model=MeetingSummaryResponse)
async def summarize_from_path(request: SummarizeRequest):
    """Complete pipeline from local file path: transcribe + diarize + summarize"""
    try:
        transcript_response = transcribe_diarize_file_path(request.file_path)
        summary_response = summarize_transcript(transcript_response.segments)
        return summary_response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
class SummarizeTranscriptRequest(BaseModel):
    segments: List[SpeakerSegment]

@app.post("/v1/summarize-transcript", response_model=MeetingSummaryResponse)
async def summarize_transcript_direct(request: SummarizeTranscriptRequest):
    """Summarize already processed transcript segments"""
    try:
        summary_response = summarize_transcript(request.segments)
        return summary_response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)