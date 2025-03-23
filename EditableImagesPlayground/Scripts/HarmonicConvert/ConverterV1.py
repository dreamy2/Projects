import os
import json
import base64
import glob
import numpy as np
import librosa

# Ask the user for frame interval (in seconds) and number of harmonics per frame
try:
    frame_interval = float(input("Enter frame interval in seconds (e.g., 0.015): ").strip())
except ValueError:
    frame_interval = 0.015
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
n_fft = 4096  # A high-resolution FFT for frequency accuracy
if hop_length < 1:
    hop_length = 1

# Compute the magnitude spectrogram
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=n_fft, center=True))
freqs = fft_frequencies(sr, n_fft)
num_frames = S.shape[1]

# Use a global amplitude reference for volume normalization
global_max = np.max(S)
if global_max == 0:
    global_max = 1  # Avoid division by zero if the audio is silent

# Set a minimum decibel level (anything below will be considered 0)
min_db = -60  # You can adjust this threshold if needed

# Prepare an array to hold the harmonic pairs for each frame.
# Each frame will have num_harmonics pairs: [pitch1, volume1, pitch2, volume2, ...]
harmonic_data = np.zeros((num_frames, 2 * num_harmonics), dtype=np.float32)

for frame in range(num_frames):
    # Get the magnitude spectrum for this frame
    magnitudes = S[:, frame].copy()
    # Zero out the DC component (0 Hz) to ignore it
    magnitudes[0] = 0

    # Get indices of the top num_harmonics peaks
    if magnitudes.size < num_harmonics:
        top_indices = np.argsort(magnitudes)[-magnitudes.size:]
    else:
        top_indices = np.argpartition(magnitudes, -num_harmonics)[-num_harmonics:]
    # Sort the indices in ascending order of frequency (for consistency)
    top_indices = top_indices[np.argsort(freqs[top_indices])]

    # For each peak, record its frequency (in Hz) and normalized volume (0-1)
    for i, idx in enumerate(top_indices):
        # Frequency in Hz (raw)
        pitch = freqs[idx]
        # Convert amplitude to decibels relative to the global maximum:
        # If the magnitude is extremely low, assign 0 volume.
        if magnitudes[idx] < 1e-10:
            volume = 0.0
        else:
            # 20*log10(amplitude / global_max) gives a dB value where 0 dB is the max
            vol_db = 20 * np.log10(magnitudes[idx] / global_max)
            # Map the dB value from [min_db, 0] to a normalized range [0, 1]
            volume = (vol_db - min_db) / (0 - min_db)
            volume = np.clip(volume, 0, 1)

        harmonic_data[frame, 2 * i] = pitch
        harmonic_data[frame, 2 * i + 1] = volume

# Pack the harmonic data as binary 32-bit floats
binary_data = harmonic_data.tobytes()
# Encode the binary data as a Base64 string
b64_str = base64.b64encode(binary_data).decode('utf-8')

# Create a JSON array with the format:
# [ "filename", frame_interval, num_harmonics, "Base64EncodedBinaryData" ]
output_data = [filename, frame_interval, num_harmonics, b64_str]

# Save the output to a JSON file in the same folder as the script
base_name, _ = os.path.splitext(filename)
output_path = os.path.join(script_dir, base_name + ".json")
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file)

print(f"Saved JSON output to {output_path}")