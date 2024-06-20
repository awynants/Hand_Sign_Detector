# Hand Sign Detector
A short MediaPipe implementation to recognize various hand signs, either from still images or from a video stream

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoint](#api-endpoint)
- [Results](#results)

## Introduction
This project uses MediaPipe, a powerful library for machine learning solutions, to detect specific hand signs in real-time using a webcam. The signs detected include:
- Open Hand
- Surfing Sign
- Peace Sign
- OK Sign (custom)
- Waving detection in real time

The project includes functionality to detect these gestures in real time and provides a Flask REST API endpoint for hand sign detection on still images.

## Features
- Real-time hand sign detection using a webcam
- Detection of multiple hand signs
- REST API for image-based hand sign detection
- Visualization of detected hand signs

## Installation
To set up the project, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/awynants/Hand_Sign_Detector.git
    cd Hand_Sign_Detector
    ```

2. **Create and activate a virtual environment:**
    ```sh
    python -m venv handsign
    handsign\Scripts\activate   
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Real-Time Detection
To run the real-time hand sign detection using your webcam:
```sh
python src/detector.py
```
The tracked hand landmarks will be shown on screen, as well as text giving the current recognized gesture.

### API Endpoint
To start the Flask server and use the API endpoint:
```sh
python src/endpoint.py
```

This allows the user to pass images to be evaluated for whether they contain one of the included signs. The API uses the POST method at the '/predict' route. An example use is:
```sh
curl -X POST -F "image=@image_path" http://127.0.0.1:5000/predict
```