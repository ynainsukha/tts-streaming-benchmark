"""Objective quality metrics for clips produced by bench.py.

  Intelligibility   ASR round-trip: Whisper transcribes the clip, compared to input text
                    -> WER / CER (lower is better). Catches mispronounced numbers, names, IDs.
  Naturalness       UTMOS22 predicted MOS (1-5, English only). A neural proxy for human
                    naturalness ratings, trained on listening-test data.
  Prosody           F0 (pitch) features via Praat: pitch range, variability, final-contour
                    slope/rise (does a question actually rise?), energy variability,
                    speech rate, pause count/length. Quantifies "emotion" differences.
  Voice consistency Speaker-embedding cosine similarity to the voice's centroid across
                    all clips: does the model keep the same voice identity run to run?

Usage:
  python eval_quality.py results/<run_id>              # all clips
  python eval_quality.py results/<run_id> --max-per-cell 3 --skip utmos
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------- intelligibility ----------
def make_normalizer(lang):
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
        from whisper_normalizer.basic import BasicTextNormalizer
        return EnglishTextNormalizer() if lang == "en" else BasicTextNormalizer()
    except ImportError:
        return lambda s: " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def _register_cuda_dlls():
    """Windows: make pip-installed CUDA libs (requirements-cuda.txt) loadable without
    touching PATH by hand. CTranslate2 loads cuBLAS/cuDNN lazily at first inference,
    so this must run before the model is used."""
    if os.name != "nt":
        return []
    try:
        import nvidia
    except ImportError:
        return []
    found = []
    for base in nvidia.__path__:
        for lib in ("cublas", "cudnn"):
            d = os.path.join(base, lib, "bin")
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                found.append(lib)
    return found


class ASR:
    def __init__(self, size, device):
        if device == "cuda":
            found = _register_cuda_dlls()
            if os.name == "nt" and not {"cublas", "cudnn"} <= set(found):
                raise SystemExit("CUDA libs not found in this venv. Run: pip install -r requirements-cuda.txt "
                                 "(or drop --asr-device cuda to run on CPU).")
        from faster_whisper import WhisperModel
        # "auto" picks CUDA whenever an NVIDIA GPU exists, even without the CUDA 12 runtime
        # (cuBLAS/cuDNN) installed, so default to CPU and make GPU opt-in.
        self.m = WhisperModel(size, device=device, compute_type="int8" if device == "cpu" else "float16")

    def __call__(self, path, lang):
        # Decode with soundfile and pass the array directly: bench.py writes 16 kHz mono,
        # which is Whisper's native input, so PyAV (and its version quirks) is never touched.
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            raise ValueError(f"{path}: expected 16 kHz, got {sr}")
        segs, _ = self.m.transcribe(wav, language=lang, beam_size=5, vad_filter=False)
        return " ".join(s.text.strip() for s in segs)


def wer_cer(ref, hyp, lang):
    import jiwer
    norm = make_normalizer(lang)
    r, h = norm(ref), norm(hyp)
    if not r.strip():
        return None, None
    return jiwer.wer(r, h), jiwer.cer(r, h)


# ---------- naturalness ----------
class UTMOS:
    def __init__(self):
        import torch
        self.torch = torch
        self.m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)

    def __call__(self, wav, sr):
        with self.torch.no_grad():
            return float(self.m(self.torch.from_numpy(wav).unsqueeze(0), sr).item())


# ---------- prosody ----------
def frame_db(wav, sr, win=0.025, hop=0.010):
    n, h = int(win * sr), int(hop * sr)
    if len(wav) < n:
        return np.array([-120.0])
    frames = np.lib.stride_tricks.sliding_window_view(wav, n)[::h]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms)


def prosody(path, wav, sr, words):
    import parselmouth
    p = parselmouth.Sound(path).to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=500)
    f0, t = p.selected_array["frequency"], p.xs()
    voiced = f0 > 0
    out = {}
    if voiced.sum() > 10:
        st = 12 * np.log2(f0[voiced] / 100.0)  # semitones re 100 Hz: speaker-independent scale
        tv = t[voiced]
        med = np.median(st)
        out["f0_median_hz"] = float(np.median(f0[voiced]))
        out["f0_range_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
        out["f0_std_st"] = float(np.std(st))
        last = tv >= tv[-1] - 0.5  # final 500 ms of voiced speech
        if last.sum() > 5:
            out["final_slope_st_per_s"] = float(np.polyfit(tv[last], st[last], 1)[0])
            tail = tv >= tv[-1] - 0.15
            out["final_rise_st"] = float(np.median(st[tail]) - med)

    db = frame_db(wav, sr)
    thr = db.max() - 35
    speech = db > thr
    idx = np.where(speech)[0]
    if len(idx):
        core = speech[idx[0]: idx[-1] + 1]
        active_s = len(core) * 0.01
        pauses, run = [], 0
        for s in core:
            if not s:
                run += 1
            else:
                if run * 10 >= 150:
                    pauses.append(run * 10)
                run = 0
        out["speech_rate_wps"] = words / active_s if active_s else None
        out["pause_count"] = len(pauses)
        out["pause_total_ms"] = float(sum(pauses))
        out["pause_mean_ms"] = float(np.mean(pauses)) if pauses else 0.0
        out["energy_std_db"] = float(np.std(db[speech]))
        out["lead_silence_ms"] = float(idx[0] * 10)  # silence before speech: adds to perceived latency
    return out


# ---------- voice consistency ----------
class SpeakerEmb:
    def __init__(self):
        from resemblyzer import VoiceEncoder, preprocess_wav
        self.enc, self.pre = VoiceEncoder(verbose=False), preprocess_wav

    def __call__(self, path):
        return self.enc.embed_utterance(self.pre(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--asr-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max-per-cell", type=int, default=0, help="clips per (model,item); 0 = all")
    ap.add_argument("--skip", nargs="*", default=[], choices=["asr", "utmos", "prosody", "speaker"])
    a = ap.parse_args()

    run = Path(a.run_dir)
    rows = [json.loads(l) for l in (run / "latency.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ok")]
    corpus = {json.loads(l)["id"]: json.loads(l) for l in Path("corpus.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

    if a.max_per_cell:
        seen, kept = defaultdict(int), []
        for r in rows:
            k = (r["model"], r["item_id"], r["voice_id"])
            if seen[k] < a.max_per_cell:
                kept.append(r)
                seen[k] += 1
        rows = kept

    asr = ASR(a.asr_model, a.asr_device) if "asr" not in a.skip else None
    utmos = UTMOS() if "utmos" not in a.skip else None
    spk = SpeakerEmb() if "speaker" not in a.skip else None

    results, embs = [], {}
    for i, r in enumerate(rows):
        text = corpus[r["item_id"]]["text"]
        wav, sr = sf.read(r["wav"], dtype="float32")
        q = {"clip_id": r["clip_id"], "model": r["model"], "item_id": r["item_id"],
             "category": r["category"], "lang": r["lang"], "voice_id": r["voice_id"]}
        if asr:
            hyp = asr(r["wav"], r["lang"])
            q["asr_text"] = hyp
            q["wer"], q["cer"] = wer_cer(text, hyp, r["lang"])
        if utmos and r["lang"] == "en":
            q["utmos"] = utmos(wav, sr)
        if "prosody" not in a.skip:
            q.update(prosody(r["wav"], wav, sr, len(text.split())))
        if spk:
            embs[r["clip_id"]] = (r["voice_id"], spk(r["wav"]))
        results.append(q)
        print(f"{i+1}/{len(rows)} {r['model']:<20} {r['item_id']:<16} "
              f"wer={q.get('wer')} utmos={q.get('utmos')}")

    if embs:
        by_voice = defaultdict(list)
        for vid, e in embs.values():
            by_voice[vid].append(e)
        cent = {v: np.mean(es, axis=0) for v, es in by_voice.items()}
        for q in results:
            vid, e = embs[q["clip_id"]]
            c = cent[vid]
            q["spk_sim"] = float(np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c)))

    with open(run / "quality.jsonl", "w", encoding="utf-8") as f:
        for q in results:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {run / 'quality.jsonl'}")


if __name__ == "__main__":
    main()
