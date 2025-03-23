import os
import json
import numpy as np
import librosa
import traceback
import sys

def extract_harmonics(file_path, harmonic_count=5, hop_length=512):
    """
    Extract harmonic data from an audio file.

    Returns a *list* of frames (instead of a dict).
    Each element in the list is a list of harmonics:
        d[0] -> harmonics for frame 1
        d[1] -> harmonics for frame 2
        ...
    """
    # 1) Load the audio, preserving its native sample rate
    y, sr = librosa.load(file_path, sr=None)
    
    # 2) Separate harmonic component
    y_harmonic = librosa.effects.harmonic(y)
    
    # 3) Compute STFT
    S = np.abs(librosa.stft(y_harmonic, hop_length=hop_length))
    
    # 4) Prepare a list for per-frame data (instead of a dict)
    frames_list = []
    num_frames = S.shape[1]  # total frames
    num_bins = S.shape[0]    # frequency bins
    
    for i in range(num_frames):
        spectrum = S[:, i]
        
        # Select indices of the top harmonic_count peaks
        peak_indices = np.argsort(spectrum)[-harmonic_count:][::-1]
        
        harmonics = []
        max_amp = np.max(spectrum) if np.max(spectrum) != 0 else 1
        
        for idx in peak_indices:
            # Convert the bin index to frequency (Hz)
            freq = idx * sr / (2 * num_bins)
            
            # Format pitch & volume as strings with exactly 4 decimal places
            pitch_str = f"{freq / 440.0:.4f}"
            volume_str = f"{(spectrum[idx] / max_amp):.4f}"
            
            harmonics.append({'p': pitch_str, 'v': volume_str})
        
        # Append the harmonics for this frame to the frames_list
        frames_list.append(harmonics)
    
    return frames_list

def main():
    # Ask user for harmonic count
    default_count = 5
    try:
        user_input = input(f"How many harmonics do you want to extract? (default {default_count}): ")
        harmonic_count = int(user_input) if user_input.strip() else default_count
    except ValueError:
        harmonic_count = default_count
    
    # Ask user for how long to wait (in seconds) for the next harmonic
    default_wait = 0.05
    try:
        user_input_r = input(f"How many seconds to wait per harmonic? (default {default_wait}): ")
        wait_time = float(user_input_r) if user_input_r.strip() else default_wait
    except ValueError:
        wait_time = default_wait
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Valid audio extensions
    valid_extensions = ('.wav', '.mp3', '.ogg', '.flac')
    
    # Gather audio files in the same folder as this script
    files = [
        os.path.join(script_dir, f)
        for f in os.listdir(script_dir)
        if f.lower().endswith(valid_extensions)
    ]
    
    if not files:
        print("No audio files found in the script's folder.")
        sys.exit(0)
    
    output_data = []
    total_files = len(files)
    
    # Process each file
    for i, file_path in enumerate(files, start=1):
        print(f"\nProcessing file {i} of {total_files}: {os.path.basename(file_path)}")
        try:
            # Extract the frame data as a list
            frames_list = extract_harmonics(
                file_path,
                harmonic_count=harmonic_count,
                hop_length=512
            )
            
            # Build the final dictionary for this file
            file_dict = {
                'n': os.path.basename(file_path),  # file name
                'r': wait_time,                   # user-specified wait time
                'c': harmonic_count,              # user-specified harmonic count
                # d is now a list of frames, each frame is a list of harmonics
                'd': frames_list
            }
            output_data.append(file_dict)
            
            # Show completion percentage
            percent_complete = (i / total_files) * 100
            print(f"Completed file {i} of {total_files}. ({percent_complete:.2f}% complete)")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            # If you want to stop on error, uncomment:
            # sys.exit(1)
    
    # Write the JSON data to harmonics.txt in the same folder
    output_file = os.path.join(script_dir, "harmonics.txt")
    try:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"\nExtraction complete. Data exported to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {e}")
        traceback.print_exc()
    
    # Auto-close (exit) when done
    sys.exit(0)

if __name__ == "__main__":
    main()
