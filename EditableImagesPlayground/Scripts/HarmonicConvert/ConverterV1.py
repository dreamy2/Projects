import os
import glob
import base64
import numpy as np
import librosa

# Ask the user for frame interval and number of harmonics
try:
    frame_interval = float(input("Enter frame interval in seconds (e.g., 0.01): ").strip())
except ValueError:
    frame_interval = 0.01
try:
    num_harmonics = int(input("Enter number of harmonics per frame (e.g., 100): ").strip())
except ValueError:
    num_harmonics = 100

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
duration = librosa.get_duration(y=y, sr=sr)

# Determine STFT parameters
hop_length = int(frame_interval * sr)
n_fft = 4096  # High-resolution FFT for frequency accuracy
if hop_length < 1:
    hop_length = 1

# Compute magnitude spectrogram
S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=n_fft, center=True))
freqs = fft_frequencies(sr, n_fft)
num_frames = S.shape[1]

# Prepare an array to hold all harmonic pairs (each pair: pitch, volume)
harmonic_data = np.zeros((num_frames, 2 * num_harmonics), dtype=np.float32)

# For each frame, extract the top num_harmonics peaks (by magnitude)
for frame in range(num_frames):
    magnitudes = S[:, frame]
    # Ignore the DC component for peak selection
    magnitudes_noDC = magnitudes.copy()
    magnitudes_noDC[0] = 0
    # Get the indices of the top num_harmonics peaks
    if magnitudes_noDC.size < num_harmonics:
        top_indices = np.argsort(magnitudes_noDC)[-magnitudes_noDC.size:]
    else:
        top_indices = np.argpartition(magnitudes_noDC, -num_harmonics)[-num_harmonics:]
    # Sort indices by frequency (ascending)
    top_indices = top_indices[np.argsort(freqs[top_indices])]
    for i, idx in enumerate(top_indices):
        freq = freqs[idx]
        vol = magnitudes[idx]
        harmonic_data[frame, 2 * i] = freq
        harmonic_data[frame, 2 * i + 1] = vol

# Pack the harmonic data as binary 32-bit floats
binary_data = harmonic_data.tobytes()
# Encode the binary data as a Base64 string
b64_str = base64.b64encode(binary_data).decode('utf-8')

# Create a Lua table string with the format:
# data = { "filename", frame_interval, num_harmonics, "base64_data" }
lua_table = f'data = {{ "{filename}", {frame_interval}, {num_harmonics}, "{b64_str}" }}'

# Save the output to a text file in the same folder as the script
base_name, _ = os.path.splitext(filename)
output_path = os.path.join(script_dir, base_name + ".txt")
with open(output_path, "w", encoding="utf-8") as txt:
    txt.write(lua_table)

print(f"Saved output to {output_path}")
