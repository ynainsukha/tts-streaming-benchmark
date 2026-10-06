# ElevenLabs TTS Latency Experiment

A small Python script that measures streaming text-to-speech latency on the ElevenLabs API, comparing a high-quality model (`eleven_v4`) with a low-latency one (`eleven_v4_turbo`).

I built this to understand the trade-off at the heart of real-time voice agents: how fast audio starts versus how natural it sounds.

## What it measures

For each request, the script records two numbers:

- **Time to first audio**: how long until the first audio chunk arrives. This is what a listener actually feels in a live conversation.
- **Full clip time**: how long until the entire clip has been received.

It reuses a single HTTP session so that only the first request pays for DNS lookup and the TLS handshake, then runs each model several times so warm-up effects can be separated from steady-state latency.

## Setup

Requires Python 3.9+ and an ElevenLabs API key.

```powershell
python -m pip install -r requirements.txt

# PowerShell
$env:ELEVENLABS_API_KEY = "your-key-here"

# macOS / Linux
export ELEVENLABS_API_KEY="your-key-here"
```

Open `tts_latency.py` and set `VOICE_ID` to a voice from your ElevenLabs voice library.

## Run

```powershell
python tts_latency.py
```

Audio files are written as `quality_N.mp3` and `fast_N.mp3`.

## Results

Test input: a four-sentence customer-support reply containing an order number, a date, a city name, and a closing question. Requests were made from India over a reused HTTP session. Round 0 is excluded as warm-up.

| Model | Time to first audio (warm) | Full clip (warm) |
|---|---|---|
| `eleven_v4` | ~920–1610 ms | ~4.6–5.0 s |
| `eleven_v4_turbo` | ~350–450 ms | ~2.0–2.4 s |

## Observations

1. **Turbo is 2–3x faster to first audio, and much more consistent.** Its first-audio latency stayed within a ~100 ms band, while `eleven_v4` varied by ~700 ms. For conversational agents, predictability matters almost as much as speed, since jitter is what makes pauses feel unnatural.

2. **Streaming matters most on longer text.** On a single short sentence, first audio and full clip time were nearly identical, so streaming added little. On the four-sentence reply, `eleven_v4` took ~5 s to finish but started speaking within ~1 s.

3. **Much of the "cold start" was networking.** The first `eleven_v4` call took ~2.1 s, but the `eleven_v4_turbo` call right after it was already fast (~330 ms). The first request paid for connection setup; the next one inherited the warm connection.

4. **The quality difference was emotional tone, not pronunciation or intonation.** Both models handled the order number and the city name equally well, and both delivered the closing question with rising, question-like intonation. But the emotional colour of that question was noticeably different between the two, in a way that is easier to hear than to name.

## Takeaway

Both models were accurate; they differed in latency and in emotional delivery. That suggests a voice agent might route per turn: the faster model for informational replies ("your order ships Thursday"), and whichever model best fits the intended tone for moments where emotion matters (questions, apologies, empathy).

## Caveats

- Small sample (three warm runs per model), measured from a single location.
- Latency includes network round-trip time from India, which affects both models equally but inflates absolute numbers.
- Quality observations are subjective, based on listening to the clips side by side.

## Possible next steps

- Run more iterations and report p50/p95 instead of ranges.
- Compare against non-streaming requests to quantify the streaming benefit directly.
- Add a lowest-latency model tier (e.g. a Flash model) to the comparison.
- Prototype per-turn model routing based on the type of reply.
