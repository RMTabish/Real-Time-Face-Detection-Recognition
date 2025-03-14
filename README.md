﻿# Real-Time Face Detection & Recognition

This project implements real-time face detection and recognition using OpenCV and the `face_recognition` library. It identifies faces from a webcam feed by comparing them with pre-stored images.

---

## Features
- Detects and recognizes faces in real-time using a webcam.
- Compares detected faces with known faces stored in the `faces/` directory.
- Draws bounding boxes and labels recognized faces.
- Uses `dlib`'s HOG-based model for face detection (can be switched to CNN if using a GPU).
- Resizes frames for better performance.

## Dependencies
- ```pip install face_recognition```
- ```pip install opencv-python opencv_contrib-python```

## Usage

Create a Virtaul enviorment and run ```python match.py```
Press q to exit

##Demo 


https://github.com/user-attachments/assets/abd8c57e-4a29-476d-bbae-8a775c38e033

