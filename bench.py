"""Streaming TTS benchmark for the ElevenLabs API.

Per request it records: time to response headers, time to first audio (TTFA),
total time, audio duration, real-time factor, per-chunk arrival times, and
simulated playback stalls at several client jitter-buffer sizes.

Audio is requested as raw PCM (16 kHz, 16-bit mono) so every byte maps to an
exact audio duration: 32,000 bytes == 1 second. That is what makes the
stall simulation exact rather than approximate.

Examples:
  python bench.py --list-models
  python bench.py --suite latency --reps 20 --models eleven_v4 eleven_v4_turbo
  python bench.py --suite latency --reps 10 --conn cold
  python bench.py --suite latency --reps 5 --concurrency 4
  python bench.py --suite quality --reps 3
"""
import argparse
import json
import os
import random
import sys
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://api.elevenlabs.io/v1"
SR = 16000
BYTES_PER_SEC = SR * 2
JITTER_BUFFERS_MS = (0, 100, 250)

_local = threading.local()


def api_key():
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("Set ELEVENLABS_API_KEY first.")
    return key


def new_session():
    s = requests.Session()
    s.headers.update({"xi-api-key": api_key(), "Content-Type": "application/json"})
    return s


def warm_session():
    if not hasattr(_local, "session"):
        _local.session = new_session()
    return _local.session


def load_corpus(path, suite=None, ids=None):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    if suite:
        rows = [r for r in rows if suite in r["suites"]]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    return rows


def simulate_playback(chunks, jitter_ms):
    """chunks: [(arrival_s, audio_s)]. Playback starts jitter_ms after the first
    chunk arrives. Returns (stall_count, total_stall_ms)."""
    buf_end, stalls, stall_s = None, 0, 0.0
    for arrival, dur in chunks:
        if buf_end is None:
            buf_end = arrival + jitter_ms / 1000 + dur
        elif arrival > buf_end:
            stalls += 1
            stall_s += arrival - buf_end
            buf_end = arrival + dur
        else:
            buf_end += dur
    return stalls, stall_s * 1000


def baseline_rtt_ms(session):
    """Round trip of a cheap authenticated GET on the same connection pool.
    TTFA minus this is a rough estimate of server-side time to first audio."""
    t0 = time.perf_counter()
    r = session.get(f"{API}/user/subscription", timeout=30)
    r.content
    return (time.perf_counter() - t0) * 1000 if r.ok else None


def run_once(session, voice_id, model, item, out_dir, extra_body=None):
    url = f"{API}/text-to-speech/{voice_id}/stream"
    body = {"text": item["text"], "model_id": model, **(extra_body or {})}
    clip_id = uuid.uuid4().hex[:10]
    rec = {"clip_id": clip_id, "model": model, "voice_id": voice_id, "item_id": item["id"],
           "category": item["category"], "lang": item["lang"], "chars": len(item["text"]),
           "words": len(item["text"].split()), "ts": datetime.now(timezone.utc).isoformat()}

    t0 = time.perf_counter()
    try:
        with session.post(url, params={"output_format": "pcm_16000"}, json=body,
                          stream=True, timeout=120) as r:
            rec["headers_ms"] = (time.perf_counter() - t0) * 1000
            rec["status"] = r.status_code
            rec["request_id"] = r.headers.get("request-id") or r.headers.get("x-request-id")
            if r.status_code != 200:
                rec.update(ok=False, error=r.text[:300])
                return rec
            pcm, chunks = bytearray(), []
            for c in r.iter_content(chunk_size=None):
                if c:
                    chunks.append((time.perf_counter() - t0, len(c) / BYTES_PER_SEC))
                    pcm.extend(c)
    except requests.RequestException as e:
        rec.update(ok=False, error=str(e)[:300])
        return rec

    total = time.perf_counter() - t0
    audio_s = len(pcm) / BYTES_PER_SEC
    if not chunks or audio_s == 0:
        rec.update(ok=False, error="empty audio")
        return rec

    wav_path = out_dir / "audio" / f"{item['id']}__{model}__{clip_id}.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(bytes(pcm[: len(pcm) // 2 * 2]))

    rec.update(
        ok=True,
        ttfa_ms=chunks[0][0] * 1000,
        total_ms=total * 1000,
        audio_s=audio_s,
        rtf=total / audio_s,  # < 1 means generated faster than real time
        stream_rtf=(total - chunks[0][0]) / max(audio_s - chunks[0][1], 1e-6),
        n_chunks=len(chunks),
        first_chunk_audio_ms=chunks[0][1] * 1000,
        chunk_arrivals_ms=[round(a * 1000, 1) for a, _ in chunks],
        chunk_audio_ms=[round(d * 1000, 1) for _, d in chunks],
        wav=str(wav_path),
    )
    for j in JITTER_BUFFERS_MS:
        n, ms = simulate_playback(chunks, j)
        rec[f"stalls_j{j}"] = n
        rec[f"stall_ms_j{j}"] = ms
    return rec


def list_models():
    r = new_session().get(f"{API}/models", timeout=30)
    r.raise_for_status()
    for m in r.json():
        print(f"{m['model_id']:<32} tts={m.get('can_do_text_to_speech')}  {m.get('name')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--models", nargs="+", default=["eleven_v4", "eleven_v4_turbo"])
    ap.add_argument("--voices", nargs="+", default=[os.environ.get("ELEVENLABS_VOICE_ID", "")])
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--suite", choices=["latency", "quality"], default="latency")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--conn", choices=["warm", "cold"], default="warm",
                    help="warm: reuse one connection; cold: new session (new TCP+TLS) per request")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    ap.add_argument("--tag", default="")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    a = ap.parse_args()

    if a.list_models:
        return list_models()
    if not a.voices or not a.voices[0]:
        sys.exit("Pass --voices <id> or set ELEVENLABS_VOICE_ID.")

    items = load_corpus(a.corpus, a.suite, a.ids)
    chars = sum(len(i["text"]) for i in items) * len(a.models) * len(a.voices) * a.reps
    print(f"{len(items)} items x {len(a.models)} models x {len(a.voices)} voices x {a.reps} reps "
          f"= ~{chars:,} characters billed (before model-specific credit multipliers)")
    if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{a.tag}" if a.tag else "")
    out = Path("results") / run_id
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(vars(a), indent=2))

    # Warm-up: one discarded request per worker so connection setup is excluded from warm runs.
    if a.conn == "warm":
        def warm(_):
            run_once(warm_session(), a.voices[0], a.models[0], items[0], out)
        with ThreadPoolExecutor(a.concurrency) as ex:
            list(ex.map(warm, range(a.concurrency)))

    def task(args):
        rep, model, voice, item = args
        s = warm_session() if a.conn == "warm" else new_session()
        rec = run_once(s, voice, model, item, out)
        rec.update(rep=rep, conn=a.conn, concurrency=a.concurrency, tag=a.tag)
        return rec

    with open(out / "latency.jsonl", "w", encoding="utf-8") as f:
        for rep in range(a.reps):
            rtt = baseline_rtt_ms(warm_session())
            # Randomise order within each round so drift (time of day, server load)
            # does not systematically favour whichever model runs first.
            trials = [(rep, m, v, i) for m in a.models for v in a.voices for i in items]
            random.shuffle(trials)
            for k in range(0, len(trials), a.concurrency):
                batch = trials[k: k + a.concurrency]
                with ThreadPoolExecutor(len(batch)) as ex:
                    for rec in ex.map(task, batch):
                        rec["rtt_ms"] = rtt
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        status = (f"ttfa={rec['ttfa_ms']:6.0f}ms total={rec['total_ms']:6.0f}ms "
                                  f"rtf={rec['rtf']:.2f}") if rec["ok"] else f"ERR {rec.get('status')} {rec.get('error')}"
                        print(f"[{rep}] {rec['model']:<20} {rec['item_id']:<16} {status}")
                time.sleep(a.pause)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
