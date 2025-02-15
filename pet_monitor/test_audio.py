import alsaaudio
import time
import wave
import numpy as np

# Open the audio device
inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, device="hw:1,0")

# Set parameters
inp.setchannels(1)  # mono
inp.setrate(48000)  # sample rate
inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)  # format
inp.setperiodsize(1024)

# Record for 5 seconds
print("Recording...")
frames = []
for i in range(0, int(48000 / 1024 * 5)):
    length, data = inp.read()
    if length > 0:
        frames.append(data)
print("Done recording")

# Save to WAV file
with wave.open('test_alsaaudio.wav', 'wb') as w:
    w.setnchannels(1)
    w.setsampwidth(2)  # 2 bytes for S16_LE format
    w.setframerate(48000)
    w.writeframes(b''.join(frames))
print("Saved to test_alsaaudio.wav")
