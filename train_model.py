import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib

# --- The function we built in Step 1 & 2 ---
def preprocess_and_extract_features(file_path, fixed_duration=2.0):
    try:
        target_sr = 16000
        audio_signal, sr = librosa.load(file_path, sr=target_sr)
        
        if np.max(np.abs(audio_signal)) > 0:
            audio_signal = audio_signal / np.max(np.abs(audio_signal))
            
        target_length = int(fixed_duration * target_sr)
        if len(audio_signal) > target_length:
            audio_signal = audio_signal[:target_length]
        else:
            padding = target_length - len(audio_signal)
            audio_signal = np.pad(audio_signal, (0, padding), 'constant')

        mfccs = librosa.feature.mfcc(y=audio_signal, sr=target_sr, n_mfcc=13)
        feature_vector = np.concatenate((np.mean(mfccs, axis=1), np.std(mfccs, axis=1)))
        return feature_vector
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- New Code for Step 3: Batch Processing & Training ---
def load_dataset(speech_dir, music_dir):
    print("Extracting features from audio files... This might take a few minutes.")
    X = [] # This will hold our 26D feature vectors
    y = [] # This will hold our labels (1 for speech, 0 for music)

    # Process Speech Files (Label 1)
    for file_name in os.listdir(speech_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(speech_dir, file_name)
            features = preprocess_and_extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(1) # 1 = Speech

    # Process Music Files (Label 0)
    for file_name in os.listdir(music_dir):
        if file_name.endswith('.wav'):
            file_path = os.path.join(music_dir, file_name)
            features = preprocess_and_extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(0) # 0 = Non-speech (Music)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 1. Define your specific folder paths
    speech_folder = r"data\speech_wav"
    music_folder = r"data\music_wav"
    
    # 2. Extract features and labels from the whole dataset
    X, y = load_dataset(speech_folder, music_folder)
    print(f"\nSuccessfully loaded {len(X)} audio samples.")

    # 3. Split the data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Initialize and train the SVM Model with RBF kernel
    print("Training the SVM Model...")
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train, y_train)

    # 5. Test the model and calculate performance
    y_pred = svm_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # 6. Save the trained model to a file so we can use it in Step 4
    model_filename = "speech_detection_svm.pkl"
    joblib.dump(svm_model, model_filename)
    print(f"\nModel successfully saved as '{model_filename}'!")