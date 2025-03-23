import os
import json
import base64
import glob
import numpy as np
import librosa

# Ask the user for frame interval (in seconds) and number of harmonics per frame
try:
    frame_interval = float(input("Enter frame interval in seconds (e.g., 0.01): ").strip())
except ValueError:
    frame_interval = 0.01
try:
    num_harmonics = int(input("Enter number of harmonics per frame (e.g., 80): ").strip())
except ValueError:
    num_harmonics = 80

# Supported audio extensions
AUDIO_EXTENSIONS = [".mp3", ".wav", ".ogg"]

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

def fft_frequencies(sr, n_fft):
    # Compute frequencies for FFT bins (only positive frequencies)
    return np.linspace(0, sr/2, 1 + n_fft//2, endpoint=True)

# Find audio files in the script's directory
audio_files = [
    os.path.join(script_dir, f)
    for f in os.listdir(script_dir)
    if any(f.lower().endswith(ext) for ext in AUDIO_EXTENSIONS)
]
if not audio_files:
    print("No audio files found in the script's directory.")
    exit(0)

# Process the first audio file found
filepath = audio_files[0]
filename = os.path.basename(filepath)
print(f"Processing {filename}...")

# Load audio (mono)
y, sr = librosa.load(filepath, sr=None, mono=True)

# Determine STFT parameters
hop_length = int(frame_interval * sr)
n_fft = 4096  # Use a high-resolution FFT for frequency accuracy
if hop_length < 1:
    hop_length = 1

# Compute magnitude spectrogram
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=n_fft, center=True))
freqs = fft_frequencies(sr, n_fft)
num_frames = S.shape[1]

# Prepare an array to hold all harmonic pairs (each pair: pitch, volume)
# Shape: (num_frames, 2 * num_harmonics)
harmonic_data = np.zeros((num_frames, 2 * num_harmonics), dtype=np.float32)

for frame in range(num_frames):
    # Get the spectrum for this frame
    magnitudes = S[:, frame].copy()
    # Determine the maximum amplitude in the frame for normalization (avoid division by zero)
    max_amp = np.max(magnitudes)
    if max_amp == 0:
        max_amp = 1

    # Ignore the DC component (0 Hz) for peak selection
    magnitudes[0] = 0

    # Get the indices of the top num_harmonics peaks
    if magnitudes.size < num_harmonics:
        top_indices = np.argsort(magnitudes)[-magnitudes.size:]
    else:
        top_indices = np.argpartition(magnitudes, -num_harmonics)[-num_harmonics:]
    # Sort indices by frequency (ascending order)
    top_indices = top_indices[np.argsort(freqs[top_indices])]

    # For each selected peak, calculate pitch and normalized volume
    for i, idx in enumerate(top_indices):
        # Convert frequency to playback speed multiplier (e.g., 440 Hz -> 1.0)
        pitch = freqs[idx] / 440.0
        # Normalize volume (0 to 1) based on the max amplitude in this frame
        volume = magnitudes[idx] / max_amp
        harmonic_data[frame, 2 * i] = pitch
        harmonic_data[frame, 2 * i + 1] = volume

# Pack the harmonic data as binary 32-bit floats
binary_data = harmonic_data.tobytes()
# Encode the binary data as a Base64 string
b64_str = base64.b64encode(binary_data).decode('utf-8')

# Create a JSON array with the format:
# [ "filename", frame_interval, num_harmonics, "base64_data" ]
output_data = [filename, frame_interval, num_harmonics, b64_str]

# Save the output to a JSON file in the same folder as the script
base_name, _ = os.path.splitext(filename)
output_path = os.path.join(script_dir, base_name + ".json")
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file)

print(f"Saved JSON output to {output_path}")