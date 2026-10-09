import time
import wave
from pathlib import Path
from faster_whisper import WhisperModel
from piper import PiperVoice

ROOT_DIR = Path(__file__).resolve().parent.parent

TEST_PHRASES = [
    "a red car at cam landscape",
    "a bus at landscape two",
    "a person with a laptop",
    "where is the juice box",
    "a bluetooth speaker on the table",
    "find a phone charger near the computer monitor"
]

def generate_test_audio(phrase: str, out_path: Path, voice: PiperVoice):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav_file:
        voice.synthesize_wav(phrase, wav_file)

def get_audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()

def main():
    print("=" * 80)
    print("ASR MODEL BENCHMARK ON CPU (int8)")
    print("=" * 80)

    # 1. Generate synthetic test audio using Piper
    model_path = ROOT_DIR / "weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx"
    config_path = ROOT_DIR / "weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx.json"
    voice = PiperVoice.load(str(model_path), config_path=str(config_path), use_cuda=False)

    audio_files = []
    scratch_dir = ROOT_DIR / "scratch/asr_bench"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(TEST_PHRASES):
        fpath = scratch_dir / f"bench_{i}.wav"
        generate_test_audio(p, fpath, voice)
        dur = get_audio_duration(fpath)
        audio_files.append((p, fpath, dur))
        print(f"Phrase {i+1} ({dur:.2f}s): '{p}'")

    initial_prompt = "person, car, bus, truck, laptop, computer monitor, juice box, soda can, bluetooth speaker, phone charger, cam_landscape, cam_landscape2."

    models_to_test = ["tiny.en", "base.en", "small.en"]
    results = {}

    for m_name in models_to_test:
        print(f"\n--- Testing faster-whisper '{m_name}' (device=cpu, compute_type=int8) ---")
        t_load_0 = time.perf_counter()
        model = WhisperModel(m_name, device="cpu", compute_type="int8")
        load_time = time.perf_counter() - t_load_0
        print(f"Loaded in {load_time:.2f}s")

        latencies = []
        transcripts = []

        # Warmup
        model.transcribe(str(audio_files[0][1]), beam_size=1)

        for expected, fpath, dur in audio_files:
            t0 = time.perf_counter()
            segments, _ = model.transcribe(str(fpath), beam_size=1, initial_prompt=initial_prompt, language="en")
            hyp = " ".join(s.text.strip() for s in segments).strip()
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed)
            transcripts.append((expected, hyp, elapsed))
            print(f"  [{elapsed:6.1f}ms] Exp: '{expected}' | Hyp: '{hyp}'")

        avg_lat = sum(latencies) / len(latencies)
        results[m_name] = {
            "load_time_s": load_time,
            "avg_latency_ms": avg_lat,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "samples": transcripts
        }

    print("\n" + "=" * 80)
    print("ASR BENCHMARK SUMMARY TABLE:")
    print(f"{'Model':<12} | {'Load (s)':<10} | {'Avg Latency (ms)':<18} | {'Min (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 80)
    for m, r in results.items():
        print(f"{m:<12} | {r['load_time_s']:<10.2f} | {r['avg_latency_ms']:<18.1f} | {r['min_latency_ms']:<10.1f} | {r['max_latency_ms']:<10.1f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
