import os
import json
import numpy as np
import librosa

def extract_harmonics(audio_file, harmonic_count, intensity, target_fps=30, n_fft=2048, hop_length=512):
    """
    Extract harmonic data from an audio file.
    
    Parameters:
      audio_file (str): Path to the audio file.
      harmonic_count (int): Maximum number of harmonics to store per frame. If intensity==0, harmonic_count is ignored.
      intensity (float): 0 to 100. 0 stores all harmonics; 100 stores only those harmonics with amplitude equal to the max.
      target_fps (float): How many frames per second to sample.
      n_fft (int): FFT window size.
      hop_length (int): Hop length for STFT.
    
    Returns:
      dict: Keys are frame indices (as strings) and values are lists of harmonics. Each harmonic is a dict with:
            "p": Roblox pitch (frequency/440, rounded to 2 decimals)
            "v": volume (amplitude, rounded to 4 decimals)
    """
    # Load audio (mono, native sample rate)
    y, sr = librosa.load(audio_file, sr=None, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    total_frames = S.shape[1]
    stft_fps = sr / hop_length
    skip = max(int(round(stft_fps / target_fps)), 1)
    freq_res = sr / n_fft

    details = {}
    frame_index = 0
    for i in range(0, total_frames, skip):
        frame_mag = S[:, i]
        # Avoid division by zero if frame is silent
        max_mag = np.max(frame_mag)
        if max_mag == 0:
            rel_values = np.zeros_like(frame_mag)
        else:
            rel_values = frame_mag / max_mag

        if intensity > 0:
            # Only store harmonics with relative amplitude >= (intensity/100)
            threshold = intensity / 100.0
            indices = [idx for idx, rel in enumerate(rel_values) if rel >= threshold]
            # If harmonic_count is > 0, take only the top N (by magnitude) of these indices
            if harmonic_count > 0 and len(indices) > harmonic_count:
                indices = sorted(indices, key=lambda idx: frame_mag[idx], reverse=True)[:harmonic_count]
        else:
            # intensity==0: store all harmonics (ignores harmonic_count)
            indices = range(len(frame_mag))
        
        harmonics = []
        for idx in indices:
            # Convert frequency (Hz) to Roblox pitch (pitch=frequency/440)
            pitch = round((idx * freq_res) / 440, 2)
            volume = round(float(frame_mag[idx]), 4)
            harmonics.append({"p": pitch, "v": volume})
        details[str(frame_index)] = harmonics
        frame_index += 1
    return details

def main():
    try:
        harmonic_count = int(input("Enter the maximum number of harmonics to extract (use 0 to ignore count and store all based on intensity): "))
    except:
        harmonic_count = 0
    try:
        intensity = float(input("Enter the intensity (0 for all harmonics, 100 for only critical harmonics): "))
    except:
        intensity = 0.0
    try:
        target_fps = float(input("Enter the target FPS for harmonic frames (e.g., 30): "))
    except:
        target_fps = 30

    # Acceptable audio file extensions
    audio_exts = {'.wav', '.mp3', '.ogg'}
    script_dir = os.path.dirname(os.path.realpath(__file__))
    audio_files = [f for f in os.listdir(script_dir)
                   if os.path.splitext(f)[1].lower() in audio_exts]

    if not audio_files:
        print("No audio files found in the script directory.")
        return

    output_data = {}
    for file in audio_files:
        audio_path = os.path.join(script_dir, file)
        print(f"Processing {file} ...")
        details = extract_harmonics(audio_path, harmonic_count, intensity, target_fps=target_fps)
        output_data[file] = {"harmonicCount": harmonic_count, "details": details}
    
    output_filename = os.path.join(script_dir, "harmonics.txt")
    with open(output_filename, "w") as out:
        # Using compact JSON separators to minimize file size.
        json.dump(output_data, out, separators=(',', ':'), ensure_ascii=False)
    
    print(f"Harmonic data saved to {output_filename}")

if __name__ == '__main__':
    main()
