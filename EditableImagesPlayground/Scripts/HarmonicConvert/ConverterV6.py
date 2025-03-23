import os
import json
import base64
import glob
import numpy as np
import librosa

# Ask the user for frame interval (seconds) and number of harmonics per frame
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
    # Returns frequency bins for the STFT (only positive frequencies)
    return np.linspace(0, sr/2, 1 + n_fft//2, endpoint=True)

def quadratic_interpolation(mag, i):
    """
    Given the magnitude spectrum array and index i of the peak,
    perform quadratic interpolation to refine the peak's index.
    Returns the offset to add to i (can be negative).
    """
    if i <= 0 or i >= len(mag) - 1:
        return 0.0
    # magnitudes at the bin before, at, and after the peak
    left = mag[i - 1]
    center = mag[i]
    right = mag[i + 1]
    denominator = (left - 2 * center + right)
    if denominator == 0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return offset

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

# Separate harmonic component for cleaner pitch tracking
y_harm = librosa.effects.harmonic(y)

# Set STFT parameters
# Increase n_fft for higher frequency resolution
n_fft = 8192  
hop_length = int(frame_interval * sr)
if hop_length < 1:
    hop_length = 1

# Compute the magnitude spectrogram of the harmonic component
S = np.abs(librosa.stft(y_harm, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, center=True))
num_frames = S.shape[1]
freqs = fft_frequencies(sr, n_fft)  # frequency bins

# Prepare an array for harmonic data: each frame has num_harmonics pairs [pitch, volume]
harmonic_data = np.zeros((num_frames, 2 * num_harmonics), dtype=np.float32)

for frame in range(num_frames):
    # Get the magnitude spectrum for this frame and copy it
    magnitudes = S[:, frame].copy()
    # For normalization, get the max amplitude in this frame (avoid division by zero)
    max_amp = np.max(magnitudes)
    if max_amp == 0:
        max_amp = 1
    # Ignore the DC component (0 Hz)
    magnitudes[0] = 0

    # Get indices of the top num_harmonics peaks
    if magnitudes.size < num_harmonics:
        peak_indices = np.argsort(magnitudes)[-magnitudes.size:]
    else:
        peak_indices = np.argpartition(magnitudes, -num_harmonics)[-num_harmonics:]
    # Sort these indices in ascending order of frequency for consistency
    peak_indices = peak_indices[np.argsort(freqs[peak_indices])]

    # Process each selected peak
    for i, idx in enumerate(peak_indices):
        # Refine peak frequency using quadratic interpolation
        offset = quadratic_interpolation(magnitudes, idx)
        refined_index = idx + offset
        # Convert refined index to frequency (Hz)
        refined_freq = refined_index * sr / n_fft
        # Convert to a pitch multiplier so that 440 Hz corresponds to 1.0
        pitch = refined_freq / 440.0
        # Normalize volume for this harmonic in the frame
        volume = magnitudes[idx] / max_amp

        harmonic_data[frame, 2 * i] = pitch
        harmonic_data[frame, 2 * i + 1] = volume

# Pack the harmonic data as binary (32-bit floats) and encode as Base64
binary_data = harmonic_data.tobytes()
b64_str = base64.b64encode(binary_data).decode('utf-8')

# Create a JSON array in the format:
# [ "filename", frame_interval, num_harmonics, "Base64EncodedBinaryData" ]
output_data = [filename, frame_interval, num_harmonics, b64_str]

# Save the output JSON file to the same directory as the script
base_name, _ = os.path.splitext(filename)
output_path = os.path.join(script_dir, base_name + ".json")
with open(output_path, "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file)

print(f"Saved JSON output to {output_path}")