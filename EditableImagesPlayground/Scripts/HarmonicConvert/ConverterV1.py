import os
import json
import numpy as np
import librosa
import traceback
import sys

def extract_harmonics(file_path, harmonic_count=5):
    """
    Extracts harmonic data from an audio file:
      - Loads the audio file using its native sampling rate.
      - Separates the harmonic component with librosa.effects.harmonic.
      - Computes STFT and, for each frame, selects the top harmonic_count peaks.
      - For each harmonic, computes a normalized pitch (frequency/440)
        and normalized volume (0..1), both rounded to 4 decimals.
      
    Returns a dict:
      {
        'n': file name,
        'r': sample rate,
        'c': harmonic_count,
        'd': {
          frame_number (as string): [
            {'p': "0.1234", 'v': "0.5678"},  # pitch, volume as strings
            ...
          ],
          ...
        }
      }
    """
    # Load the audio file (sr=None to preserve original sample rate)
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract harmonic component
    y_harmonic = librosa.effects.harmonic(y)
    
    # Compute STFT (Short-Time Fourier Transform) of the harmonic signal
    S = np.abs(librosa.stft(y_harmonic))
    
    # Prepare the structure for per-frame data
    frame_data = {}
    num_frames = S.shape[1]
    num_bins = S.shape[0]
    
    for i in range(num_frames):
        spectrum = S[:, i]
        
        # Select indices of the top harmonic_count peaks in this frame
        peak_indices = np.argsort(spectrum)[-harmonic_count:][::-1]
        
        harmonics = []
        # Avoid division by zero if the frame is silent
        max_amp = np.max(spectrum) if np.max(spectrum) != 0 else 1
        
        for idx in peak_indices:
            # Convert the bin index to frequency (Hz)
            freq = idx * sr / (2 * num_bins)
            
            # Format pitch and volume as strings with exactly 4 decimal places
            pitch_str = f"{freq / 440.0:.4f}"
            volume_str = f"{(spectrum[idx] / max_amp):.4f}"
            
            harmonics.append({'p': pitch_str, 'v': volume_str})
        
        # Store the harmonics for this 1-indexed frame
        frame_data[str(i + 1)] = harmonics
    
    return {
        'n': os.path.basename(file_path),
        'r': sr,  # sample rate (int)
        'c': harmonic_count,
        'd': frame_data
    }

def main():
    # Ask the user how many harmonics to extract
    default_count = 5
    try:
        user_input = input(f"How many harmonics do you want to extract? (default {default_count}): ")
        harmonic_count = int(user_input) if user_input.strip() else default_count
    except ValueError:
        harmonic_count = default_count
    
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
            # Extract harmonics with the user-specified harmonic_count
            harmonic_info = extract_harmonics(file_path, harmonic_count=harmonic_count)
            output_data.append(harmonic_info)
            
            # Show completion percentage
            percent_complete = (i / total_files) * 100
            print(f"Completed file {i} of {total_files}. ({percent_complete:.2f}% complete)")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
            # Decide if you want to stop or continue on error
            # sys.exit(1)  # <-- uncomment if you want to stop on the first error
    
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
