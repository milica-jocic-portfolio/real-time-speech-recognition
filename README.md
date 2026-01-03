Real-Time Serbian Speech Recognition

This project implements a real-time speech recognition pipeline for Serbian using a pretrained wav2vec2 CTC model.
It captures microphone audio, segments speech using RMS-based thresholds, performs transcription, and forwards recognized text to a local REST service.

The system is designed to be low-latency, stream-based, and easy to extend.

Features
  Real-time microphone audio capture (sounddevice)
  RMS-based speech segmentation (simple VAD-like logic)
  Asynchronous audio processing using a worker thread
  Speech-to-text transcription with wav2vec2-xls-r-juznevesti-sr
  Normalization and filtering of recognized characters
  Optional forwarding of recognized text to a local HTTP service

How It Works (High Level)

1. Audio is captured from the microphone in fixed-size frames.
2. RMS energy is computed for each frame.
3. When energy exceeds a start threshold, speech recording begins.
4. Speech ends after sustained silence.
5. The collected audio segment is sent to a background worker thread.
6. The worker performs ASR inference using a pretrained wav2vec2 CTC model.
7. The decoded text is normalized and printed.
8. The result is optionally sent to a local REST endpoint.
This approach avoids blocking the audio callback and keeps latency low.

Model

Model: classla/wav2vec2-xls-r-juznevesti-sr
Framework: Hugging Face Transformers
Type: Pretrained CTC ASR model for Serbian
Model weights are not stored in this repository and are downloaded automatically at runtime.

Requirements

  Python 3.9+
Main dependencies:
  sounddevice
  numpy
  torch
  transformers
  requests
Install dependencies with: pip install -r requirements.txt

Running the Project

1. Make sure you have a working microphone.
2. (Optional) Start a local service listening at: http://localhost:5000/transcribe
3. Run the script: python main.py
4. Speak into the microphone.
5. Press Enter to stop recording.
Recognized text will be printed to the console in real time.

Configuration

Key parameters can be adjusted in the code:
  Sample rate and frame size
  RMS thresholds for speech start/end
  Minimum speech duration
  Maximum silence duration
  REST service URL
These values are heuristic and can be tuned for different environments.

Limitations
This is not a full VAD implementation (no neural or WebRTC VAD).
Speech segmentation is based on simple RMS thresholds.
No language model or beam search decoding is used.
Performance may degrade in noisy environments.
The REST service is assumed to exist but is not included.

Intended Use
  Educational purposes
  Audio / speech processing experimentation
  Prototyping real-time ASR systems
It is not production-ready without further robustness improvements.

License
This project uses third-party pretrained models subject to their respective licenses.
Refer to the Hugging Face model page for licensing details.
