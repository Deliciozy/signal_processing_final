import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def create_visualization(speech_path, music_path):
    # Load one sample of each
    y_speech, sr = librosa.load(speech_path, sr=16000, duration=2.0)
    y_music, _ = librosa.load(music_path, sr=16000, duration=2.0)

    plt.figure(figsize=(12, 10))

    # --- 1. Waveforms --- [cite: 118, 158]
    plt.subplot(2, 2, 1)
    librosa.display.waveshow(y_speech, sr=sr, color='blue')
    plt.title('Speech Waveform (Rhythmic/Vocal)')
    
    plt.subplot(2, 2, 2)
    librosa.display.waveshow(y_music, sr=sr, color='orange')
    plt.title('Music Waveform (Continuous/Steady)')

    # --- 2. Spectrograms (STFT) --- [cite: 8, 131, 177]
    # Speech Spectrogram
    plt.subplot(2, 2, 3)
    D_speech = librosa.amplitude_to_db(np.abs(librosa.stft(y_speech)), ref=np.max)
    librosa.display.specshow(D_speech, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Speech Spectrogram (Sparse/Formants)')

    # Music Spectrogram
    plt.subplot(2, 2, 4)
    D_music = librosa.amplitude_to_db(np.abs(librosa.stft(y_music)), ref=np.max)
    librosa.display.specshow(D_music, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Music Spectrogram (Dense/Harmonic)')

    plt.tight_layout()
    plt.savefig('project_visualizations.png') # Saves the image for your report
    plt.show()

# Update these with actual files from your GTZAN dataset
# Use the actual file names you see in your folders
speech_file = r"data\speech_wav\god.wav" # <-- Change this name
music_file = r"data\music_wav\ballad.wav"   # <-- Change this name

if os.path.exists(speech_file) and os.path.exists(music_file):
    create_visualization(speech_file, music_file)
else:
    print("Check your file paths to ensure the .wav files exist!")