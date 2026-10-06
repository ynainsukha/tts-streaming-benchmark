import os
import time
import requests

API_KEY = os.environ["ELEVENLABS_API_KEY"]
VOICE_ID = os.environ["CUSTOMER_CARE_VOICE_ID"]
TEXT = (
    "Hi Priya, thanks for reaching out about your order. "
    "I can see that order number 48217 was shipped on October 3rd and is currently "
    "with our delivery partner in Ahmedabad. It should reach you by Thursday evening, "
    "and you'll get an SMS with the tracking link once it's out for delivery. "
    "Is there anything else I can help you with today?"
)
SESSION = requests.Session()

def stream_tts(model_id: str, out_file: str) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    body = {"text": TEXT, "model_id": model_id}

    start = time.perf_counter()
    first_chunk = None

    with SESSION.post(url, json=body, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(out_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    if first_chunk is None:
                        first_chunk = time.perf_counter() - start
                    f.write(chunk)

    total = time.perf_counter() - start
    print(f"{model_id}: first audio in {first_chunk*1000:.0f} ms, "
          f"full clip in {total*1000:.0f} ms -> {out_file}")

if __name__ == "__main__":
    for i in range(4):
        stream_tts("eleven_v4", f"quality_{i}.mp3")
        stream_tts("eleven_v4_turbo", f"fast_{i}.mp3")