# Voice-Aware Background Music Controller

## Project Overview

This project implements a real-time voice detection system that automatically adjusts background music volume when human speech is detected.

The system listens to microphone input and uses a trained machine learning model to detect whether a person is speaking. When speech is detected, the background music volume is automatically reduced. When speech stops, the music volume returns to its original level.

This simulates a common audio mixing feature used in streaming, broadcasting, and smart audio systems.

---

## How It Works

The system works in three main steps:

1. **Feature Extraction**
   Audio features are extracted from microphone input.

2. **Voice Detection Model**
   A machine learning classifier is trained to distinguish between speech and non-speech audio.

3. **Dynamic Volume Control**

* Speech detected → background music volume reduced to **30%**
* Speech stops → background music volume returns to **100%**

---

## Repository Structure

```
3_live_demo.py              # Real-time demo program
train_model.py              # Script used to train the voice detection model
visualize_features.py       # Feature visualization for audio data
requirements.txt            # Required Python libraries
project_visualizations.png  # Visualization of extracted features
README.md                   # Project documentation
```

---

## Dataset

The dataset used for training contains speech and non-speech audio samples.

Due to GitHub file size limitations, the dataset is **not included in this repository**.
The dataset used for training is approximately **330MB**, which exceeds GitHub's file upload limit.


---

## Installation

Install the required Python libraries:

```
pip install -r requirements.txt
```

---

## Training the Model

To train the speech detection model, run:

```
python train_model.py
```

This script will load the dataset from the `data/` directory, extract audio features, and train a classifier.

---

## Running the Live Demo

To run the real-time voice detection demo:

```
python 3_live_demo.py
```

The program will:

1. Ask the user to input the name of a background music file
2. Monitor microphone input
3. Detect when speech occurs
4. Automatically adjust the background music volume

---

## Example Behavior

* Background music starts at **100% volume**
* When the user speaks → music volume decreases to **30%**
* When the user stops speaking → music volume returns to **100%**

---

## Requirements

Main libraries used in this project:

* Python
* NumPy
* Librosa
* Scikit-learn
* PyAudio
* Pygame

All dependencies are listed in `requirements.txt`.

---

## Project Goal

The goal of this project is to demonstrate how machine learning can be applied to real-time audio processing to automatically adjust background music based on human speech detection.
