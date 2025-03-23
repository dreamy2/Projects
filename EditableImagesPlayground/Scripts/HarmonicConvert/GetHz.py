import numpy as np
import wave
import os

def create_tone(filename, freq, duration=1.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)  # amplitude 0.5 to avoid clipping
    waveform_integers = np.int16(waveform * 32767)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    with wave.open(filepath, 'w') as wav_file:
        n_channels = 1
        sampwidth = 2
        wav_file.setparams((n_channels, sampwidth, sample_rate, 0, 'NONE', 'not compressed'))
        wav_file.writeframes(waveform_integers.tobytes())
    print(f"Created {filepath}")

def main():
    frequencies = [220, 880, 1760]
    for freq in frequencies:
        filename = f"tone_{freq}.wav"
        create_tone(filename, freq)

if __name__ == '__main__':
    main()