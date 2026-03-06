import sounddevice as sd
import numpy as np
import librosa
import pygame
import time
import joblib
import warnings

# Ignore librosa/numba warnings to keep the console clean for your demo
warnings.filterwarnings("ignore")

# ==========================================================
# 1. CONFIGURATION (CHANGE THESE TO MATCH YOUR PROJECT)
# ==========================================================
# Load My BGM
my_bgm = input("Enter the path to your background music file (e.g., 'bgm.mp3'): ")
BGM_FILE = "data/music_wav/"+my_bgm
# Load My Trained AI Model
MODEL_FILE = "speech_detection_svm.pkl"
try:
    svm_model = joblib.load(MODEL_FILE)
    print(f"Successfully loaded {MODEL_FILE}!")
except FileNotFoundError:
    print("Error: Could not find the trained model. Make sure it's in the same folder.")
    exit()

SAMPLE_RATE = 16000               # Must match what you used for training
BLOCK_SIZE = 8000                 # 0.5 second chunks (16000 * 0.5)
CHANNELS = 1                      # Mono mic

# Volume Logic
NORMAL_VOL = 1.0                  # 100%
DUCKED_VOL = 0.3                  # 30%
HOLD_TIME = 0.8                   # Seconds to wait before volume goes back up

# ==========================================================
# 2. LOAD YOUR AI MODEL
# ==========================================================
try:
    model = joblib.load(MODEL_FILE)
    print(f"✅ Model '{MODEL_FILE}' loaded successfully.")
except:
    model = None
    print(f"⚠️ Could not find '{MODEL_FILE}'. Running in TEST MODE (Manual Trigger).")

# ==========================================================
# 3. AI PREDICTION LOGIC
# ==========================================================
def predict_speech(audio_chunk):
    """
    Modified to extract 26 features (13 MFCCs + 13 Deltas) 
    to match your SVC model requirements.
    """
    # Standardize data
    audio_data = audio_chunk.astype(np.float32)

    # 1. Extract the base 13 MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
    
    # 2. Extract the 13 Deltas (this makes 26 total)
    # This measures how the MFCCs change over time
    deltas = librosa.feature.delta(mfccs)
    
    # 3. Combine them (Stack them on top of each other)
    # We take the mean of both to get a single 26-element vector
    mfcc_mean = np.mean(mfccs.T, axis=0)
    delta_mean = np.mean(deltas.T, axis=0)
    
    # Final feature vector: 13 + 13 = 26
    features = np.hstack([mfcc_mean, delta_mean]).reshape(1, -1)

    # IF MODEL EXISTS: Use AI to predict
    if model is not None:
        prediction = model.predict(features)
        # We changed the 1 to a 0 here!
        return True if prediction[0] == 0 else False
    
    return False

# ==========================================================
# 4. REAL-TIME SYSTEM ENGINE
# ==========================================================
last_speech_time = 0
is_ducking = False

def audio_callback(indata, frames, time_info, status):
    """This runs every time the mic hears a chunk of sound."""
    global last_speech_time, is_ducking
    
    if status:
        print(status)

    # Ask the AI what it heard
    speech_detected = predict_speech(indata.flatten())
    current_time = time.time()

    if speech_detected:
        last_speech_time = current_time
        if not is_ducking:
            pygame.mixer.music.set_volume(DUCKED_VOL)
            is_ducking = True
            print(">>> 🗣️ SPEECH DETECTED (Volume -> 30%)")

def main():
    global is_ducking, last_speech_time

    # Initialize Music
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(BGM_FILE)
        pygame.mixer.music.play(-1) # Loop forever
        print(f"🎵 Playing: {BGM_FILE}")
    except:
        print(f"❌ ERROR: Could not find music file '{BGM_FILE}'")
        return

    # Initialize Microphone
    print("\n--- DEMO STARTING ---")
    print("Adjusting volume in real-time. Press Ctrl+C to stop.\n")

    with sd.InputStream(samplerate=SAMPLE_RATE, 
                        channels=CHANNELS, 
                        blocksize=BLOCK_SIZE, 
                        callback=audio_callback):
        try:
            while True:
                # Logic to return volume to 100%
                now = time.time()
                if is_ducking and (now - last_speech_time > HOLD_TIME):
                    pygame.mixer.music.set_volume(NORMAL_VOL)
                    is_ducking = False
                    print("<<< 🔈 SILENCE (Volume -> 100%)")
                
                time.sleep(0.1) # Prevents CPU overload
        except KeyboardInterrupt:
            print("\nDemo stopped by user.")

    pygame.mixer.music.stop()
    pygame.quit()

if __name__ == "__main__":
    main()