import sounddevice as sd
import numpy as np
import torch
import requests
from transformers import AutoProcessor, AutoModelForCTC
from queue import Queue
from threading import Thread

SAMPLE_RATE = 16000
CHANNELS = 1

FRAME_MS = 100
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

RMS_START = 0.015
RMS_END = 0.008

MIN_SPEECH_SEC = 0.5
MIN_SAMPLES = int(MIN_SPEECH_SEC * SAMPLE_RATE)

MAX_SILENCE_SEC = 0.8
MAX_SILENCE_FRAMES = int(MAX_SILENCE_SEC * 1000 / FRAME_MS)

RMS_THRESHOLD = 0.005

SERVICE_URL = "http://localhost:5000/transcribe"

print("Loading model...")
processor = AutoProcessor.from_pretrained("classla/wav2vec2-xls-r-juznevesti-sr")
model = AutoModelForCTC.from_pretrained("classla/wav2vec2-xls-r-juznevesti-sr")
model.eval()
print("Model loaded")

#buffers
speech_buffer = []
in_speech = False
silence_frames = 0

processing_queue = Queue()

#normalization
ALLOWED = set("abcdefghijklmnopqrstuvwxyzšđčćž ")


def normalize_letters(text):
    text = text.lower()
    return "".join(c for c in text if c in ALLOWED)


#service connection
def send_to_service(letters):
    if not letters.strip():
        return
    try:
        requests.post(
            SERVICE_URL,
            json={"letters": letters},
            timeout=1.0
        )
    except Exception:
        pass


#speech procesing
def process_speech_worker():
    while True:
        audio = processing_queue.get()
        if audio is None:  #stop sign
            break

        if len(audio) < MIN_SAMPLES:
            continue

        #checking the average RMS of the entire audio segment
        overall_rms = np.sqrt(np.mean(audio ** 2))
        if overall_rms < RMS_THRESHOLD:
            continue

        try:
            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            raw_text = processor.decode(predicted_ids[0])

            letters = normalize_letters(raw_text)

            if letters:
                print(letters, flush=True)
                send_to_service(letters)
        except Exception as e:
            print(f"Processing error: {e}")


#voice recording function
def audio_callback(indata, frames, time_info, status):
    global in_speech, silence_frames, speech_buffer

    if status:
        if 'overflow' not in str(status).lower():
            print(status)

    frame = indata[:, 0].copy()
    rms = np.sqrt(np.mean(frame ** 2))

    if in_speech:
        speech_buffer.extend(frame)

        if rms < RMS_END:
            silence_frames += 1
            if silence_frames >= MAX_SILENCE_FRAMES:
                audio_to_process = np.array(speech_buffer, dtype=np.float32)
                processing_queue.put(audio_to_process)

                speech_buffer = []
                in_speech = False
                silence_frames = 0
        else:
            silence_frames = 0
    else:
        if rms > RMS_START:
            in_speech = True
            silence_frames = 0
            speech_buffer.extend(frame)



worker_thread = Thread(target=process_speech_worker, daemon=True)
worker_thread.start()

#poziv
print("Recording started. Press Enter to exit.")

try:
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=audio_callback,
        latency='low'
    )

    stream.start()
    input()
except KeyboardInterrupt:
    print("\nInterruption...")
finally:
    stream.stop()
    stream.close()
    processing_queue.put(None)
    worker_thread.join(timeout=2)
    print("Recording stopped.")