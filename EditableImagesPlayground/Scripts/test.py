import os
import subprocess
import json
import heapq
import numpy as np
import scipy.io.wavfile as wavfile

##############################################################################
# USER-ADJUSTABLE PARAMETERS
##############################################################################
TOP_N = 10        # Number of strongest partials to keep per frame
FRAME_SIZE = 1024 # Samples per STFT frame
HOP_SIZE = 512    # Step size between frames (overlap = FRAME_SIZE - HOP_SIZE)
##############################################################################

def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Uses ffmpeg (via subprocess) to convert an MP3 file to WAV format.
    Requires ffmpeg to be installed and in PATH.
    """
    # Example command: ffmpeg -i input.mp3 output.wav
    # The 'check=True' argument will raise an error if ffmpeg fails.
    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True)

def get_top_partials(wav_path, top_n=10, frame_size=1024, hop_size=512):
    """
    Reads a WAV file, performs STFT, and extracts the top N frequency
    components (partials) in each time frame.

    Returns a list of frames, where each frame is a list of tuples:
      (frequency_in_Hz, amplitude, phase_in_radians)
    """
    # Read the WAV file
    sample_rate, audio_data = wavfile.read(wav_path)

    # If stereo, convert to mono (average the two channels)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Ensure float for STFT calculations
    audio_data = audio_data.astype(np.float32)

    # Calculate number of frames
    num_samples = len(audio_data)
    num_frames = (num_samples - frame_size) // hop_size + 1

    # Hanning window to reduce spectral leakage
    window = np.hanning(frame_size)

    all_frames = []

    for frame_idx in range(num_frames):
        start = frame_idx * hop_size
        end = start + frame_size

        frame = audio_data[start:end]

        # Zero-pad if we don't have enough samples at the end
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')

        # Apply window
        frame = frame * window

        # Compute the real-valued FFT (since audio is real)
        spectrum = np.fft.rfft(frame)

        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)

        # Find indices of the top N magnitudes (skip bin 0 if you want to ignore DC)
        largest_indices = heapq.nlargest(
            top_n,
            range(1, len(magnitudes)),
            key=lambda i: magnitudes[i]
        )

        # Collect partials: (frequency, amplitude, phase)
        partials = []
        for idx in largest_indices:
            freq_hz = (idx * sample_rate) / frame_size
            amp = magnitudes[idx]
            ph = phases[idx]
            partials.append((float(freq_hz), float(amp), float(ph)))

        all_frames.append(partials)

    return all_frames

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Loop through files in the same directory as this script
    for filename in os.listdir(script_dir):
        if filename.lower().endswith(".mp3"):
            mp3_path = os.path.join(script_dir, filename)
            base_name = os.path.splitext(filename)[0]

            # Construct the WAV and output text filenames
            wav_filename = base_name + ".wav"
            wav_path = os.path.join(script_dir, wav_filename)
            txt_filename = base_name + "_partials.txt"
            txt_path = os.path.join(script_dir, txt_filename)

            print(f"Converting {filename} to WAV...")
            convert_mp3_to_wav(mp3_path, wav_path)

            print(f"Extracting top partials from {wav_filename}...")
            partials_data = get_top_partials(
                wav_path,
                top_n=TOP_N,
                frame_size=FRAME_SIZE,
                hop_size=HOP_SIZE
            )

            print(f"Writing partials to {txt_filename}...")
            with open(txt_path, "w", encoding="utf-8") as f:
                json.dump(partials_data, f, indent=2)

            # (Optional) Remove the WAV file to save space:
            # os.remove(wav_path)

            print(f"Done processing {filename}!\n")

if __name__ == "__main__":
    main()
