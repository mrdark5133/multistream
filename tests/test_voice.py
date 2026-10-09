"""Tests for MULTIStream Voice Interface (ASR, TTS, templating, and API)."""
import io
import os
import numpy as np
import soundfile as sf
import pytest
from starlette.testclient import TestClient

from src.voice.camera_match import match_spoken_to_camera, normalize_query_cameras
from src.voice.answers import template_answer
from src.voice.asr import WhisperASR, build_initial_prompt
from src.voice.tts import PiperTTS
from src.api.app import app


def test_camera_fuzzy_matching():
    cams = ["cam_landscape", "mobile_cam01", "mobile_cam02", "cam_entrance"]
    
    # Exact and close fuzzy match
    assert match_spoken_to_camera("cam landscape", cams) == "cam_landscape"
    assert match_spoken_to_camera("camera landscape", cams) == "cam_landscape"
    assert match_spoken_to_camera("mobile cam 1", cams) == "mobile_cam01"
    assert match_spoken_to_camera("mobile cam 02", cams) == "mobile_cam02"
    assert match_spoken_to_camera("entrance", cams) == "cam_entrance"
    assert match_spoken_to_camera("completely unrelated sky", cams) is None


def test_normalize_query_cameras():
    q1 = "find a white car in camera landscape please"
    norm1 = normalize_query_cameras(q1)
    assert "cam_landscape" in norm1
    
    q2 = "show juice box on mobile cam one"
    norm2 = normalize_query_cameras(q2)
    assert "mobile_cam01" in norm2


def test_answer_templating():
    # 0 results
    empty_ans = template_answer({"status": "success", "results": []}, "a red car")
    assert "No objects found matching" in empty_ans
    assert "a red car" in empty_ans
    
    # 1 result
    single_res = {
        "status": "success",
        "results": [{
            "label": "car",
            "color": "red",
            "camera": "cam_landscape",
            "timestamp": "2026-10-08T14:30:00"
        }]
    }
    single_ans = template_answer(single_res, "red car")
    assert "Found a red car on landscape" in single_ans
    
    # Multiple results
    multi_res = {
        "status": "success",
        "results": [
            {
                "label": "car",
                "color": "white",
                "camera": "cam_landscape",
                "timestamp": "2026-10-08T14:35:00"
            },
            {
                "label": "car",
                "color": "red",
                "camera": "mobile_cam01",
                "timestamp": "2026-10-08T14:20:00"
            }
        ]
    }
    multi_ans = template_answer(multi_res, "car")
    assert "Found 2 matches" in multi_ans
    assert "white car on landscape" in multi_ans
    
    # Clarify
    clarify_res = {
        "status": "clarify",
        "referent": "the garage",
        "options": ["cam_landscape", "mobile_cam01"]
    }
    clarify_ans = template_answer(clarify_res, "car in the garage")
    assert "Did you mean landscape or mobile cam one?" in clarify_ans


def test_whisper_initial_prompt_contains_domain_vocabulary():
    prompt = build_initial_prompt()
    assert "cam_landscape" in prompt
    assert "mobile_cam01" in prompt
    assert "juice box" in prompt
    assert "power bank" in prompt
    assert "car" in prompt
    assert "bounding box" in prompt


@pytest.fixture(scope="module")
def tts_engine():
    return PiperTTS()


@pytest.fixture(scope="module")
def asr_engine():
    return WhisperASR()


def test_piper_tts_and_whisper_transcription(tts_engine, asr_engine):
    # Synthesize audio
    text = "Find a red car near camera landscape."
    audio_id, wav_path = tts_engine.synthesize(text)
    wav_bytes = wav_path.read_bytes()
    assert len(wav_bytes) > 1000
    
    # Verify WAV header
    buf = io.BytesIO(wav_bytes)
    audio_data, sr = sf.read(buf)
    assert sr == 16000
    assert len(audio_data) > 0
    
    # Transcribe via Whisper
    transcript, info = asr_engine.transcribe_bytes(wav_bytes)
    assert len(transcript) > 0
    assert "car" in transcript.lower()


def test_voice_api_endpoints(tts_engine):
    client = TestClient(app)
    
    # 1. Synthesize valid query
    query_text = "car in cam_landscape"
    audio_id, wav_path = tts_engine.synthesize(query_text)
    wav_bytes = wav_path.read_bytes()
    
    # POST /voice
    res = client.post(
        "/voice",
        files={"file": ("query.wav", wav_bytes, "audio/wav")}
    )
    assert res.status_code == 200
    data = res.json()
    assert "transcript" in data
    assert "spoken_text" in data
    assert "audio_url" in data
    assert "timings_ms" in data
    assert data["timings_ms"]["asr"] > 0
    assert data["timings_ms"]["tts"] >= 0
    
    # 2. Test fetching generated audio
    audio_url = data["audio_url"]
    audio_file_id = audio_url.split("/")[-1]
    res_audio = client.get(f"/voice/audio/{audio_file_id}")
    assert res_audio.status_code == 200
    assert res_audio.headers["content-type"] == "audio/wav"
    assert len(res_audio.content) > 0
    
    # 3. Test silence audio (RMS < 0.001)
    silence = np.zeros(16000, dtype=np.float32)
    silence_buf = io.BytesIO()
    sf.write(silence_buf, silence, 16000, format="WAV")
    res_silence = client.post(
        "/voice",
        files={"file": ("silence.wav", silence_buf.getvalue(), "audio/wav")}
    )
    assert res_silence.status_code == 400
    assert "No speech detected" in res_silence.json()["detail"]
    
    # 4. Test corrupt audio
    res_corrupt = client.post(
        "/voice",
        files={"file": ("corrupt.wav", b"not a wav file content", "audio/wav")}
    )
    assert res_corrupt.status_code == 400
    assert "Invalid or corrupt WAV" in res_corrupt.json()["detail"]
