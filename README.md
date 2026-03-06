# signal_processing_final# Voice-Aware Background Music Controller

## Project Overview

This project implements a real-time voice detection system that automatically adjusts background music volume based on whether human speech is detected.

The system listens to microphone input and detects the presence of human voice using a trained machine learning model. When speech is detected, the system lowers the background music volume to make the voice clearer. When speech stops, the music volume returns to its original level.

This feature simulates an automatic audio mixing behavior commonly used in streaming, broadcasting, and smart audio systems.

---

## How It Works

The system works in three main steps:

1. **Feature Extraction**
   Audio features are extracted from microphone input (such as spectral or audio signal features).

2. **Voice Detection Model**
   A machine learning model (SVM) is trained to classify whether the input audio contains human speech.

3. **Dynamic Music Volume Control**

   * If speech is detected → background music volume is reduced to **30%**
   * If speech stops → background music volume returns to **100%**

---

## Project Structure

```
project/
│
├── 3_live_demo.py              # Real-time demo for voice detection and music control
├── train_model.py               # Script used to train the voice detection model
├── visualize_features.py        # Visualization of extracted audio features
├── requirements.txt             # Python dependencies
├── speech_detection_svm.pkl     # Trained machine learning model
├── project_visualizations.png   # Feature visualization results
└── data/                        # Dataset used for training
```

---

## Dataset

The training dataset was obtained from publicly available audio datasets containing speech and non-speech samples.

The dataset is used to train a classifier that distinguishes between human voice and background noise/music.

---

## Installation

Clone the repository and install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the speech detection model, run:

```bash
python train_model.py
```

This script will process the dataset, extract audio features, and train an SVM classifier.

---

## Running the Live Demo

To run the real-time system:

```bash
python 3_live_demo.py
```

The program will:

1. Load the trained model
2. Ask for a background music file name
3. Monitor microphone input
4. Adjust music volume automatically when speech is detected

---

## Example Behavior

* Background music starts playing at **100% volume**
* When the user speaks into the microphone → music volume reduces to **30%**
* When the user stops speaking → music returns to **100% volume**

---

## Requirements

Main libraries used in this project include:

* Python
* NumPy
* Librosa
* Scikit-learn
* PyAudio
* Pygame (for music playback)

All dependencies are listed in `requirements.txt`.

---

## Project Goal

The goal of this project is to demonstrate how machine learning can be used in real-time audio processing to improve user experience in applications such as streaming, smart assistants, and audio production tools.
