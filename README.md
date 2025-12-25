# 🤟 Sign Language Translation System (ASL)

A real-time **Sign Language Translation** system that recognizes **American Sign Language (ASL) hand gestures** using a webcam and converts them into readable text. The project leverages **Computer Vision, MediaPipe, and Deep Learning (MobileNetV2)** for accurate and smooth real-time predictions.

---

## 📌 Project Overview

This project captures live video from a webcam, detects hands using **MediaPipe Hands**, and classifies ASL gestures using a **MobileNetV2-based CNN model** trained on the ASL Alphabet dataset. Predictions are stabilized using a rolling buffer and displayed as a sentence in real time.

---

## ✨ Features

- 🎥 Real-time webcam-based ASL recognition  
- ✋ Hand detection using MediaPipe  
- 🧠 Deep learning with MobileNetV2  
- 🔄 Prediction smoothing with deque buffer  
- 📝 Live sentence formation  
- ⚡ Efficient and lightweight for real-time use  

---

## 🧠 Model Details

- Backbone: MobileNetV2 (ImageNet pretrained)  
- Input Size: 224 × 224 × 3  
- Layers:
  - Data Augmentation
  - Global Average Pooling
  - Dropout (0.3)
  - Dense Softmax Output  
- Training:
  - Transfer Learning
  - Fine-tuning last 30 layers  
- Number of Classes: 29 (ASL signs)

---

## 🗂️ Project Structure
```
SignLanguage_Translation/
│
├── app.py                      # Main real-time ASL translation application
├── asl_labels.json             # Label index to ASL class mapping
├── model.h5                    # Trained deep learning model (HDF5 format)
├── model.keras                 # Trained model (Keras native format)
├── requirements.txt            # Python dependencies
├── ASL_symbols.png             # ASL alphabet reference image
├── captured_frame.jpg          # Sample captured webcam frame
├── captured_hand.jpg           # Sample cropped hand image
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
└── research/                   # Model training and experimentation
    ├── asl-translation.ipynb   # Training, preprocessing, and evaluation notebook
    └── prediction.ipynb        # Model inference and prediction testing
```
---

## 🚀 How It Works

1. Captures live video from webcam  
2. Detects hand landmarks using MediaPipe  
3. Crops and preprocesses hand region  
4. Passes image to trained CNN model  
5. Smooths predictions using buffer  
6. Forms sentence from detected signs  
7. Displays translated text in real time  

---

## 🛠️ Installation

### Clone Repository
```bash
git clone https://github.com/kishor2004reddy/SignLanguage_Translation.git  
cd SignLanguage_Translation
```

### Install Dependencies
```bash
pip install -r requirements.txt  
```

Required:
- Python 3.8+
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy

---

## ▶️ Run the Application
```cmd
python app.py  
```
Press **Q** to quit the application.

---

## 📊 Dataset

- ASL Alphabet Dataset  
- ~87,000 images  
- 29 classes  
- Used data augmentation and normalization  
- Trained on Kaggle with GPU support  

---

## 🔮 Future Enhancements

- Word and sentence-level ASL translation  
- NLP-based grammar correction  
- Text-to-speech support  
- Dynamic gesture recognition  
- Web or mobile deployment  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Yarramaddi Kishor Kumar Reddy**  
GitHub: https://github.com/kishor2004reddy  

---

## ⭐ Acknowledgements

- MediaPipe (Google)  
- TensorFlow & Keras  
- ASL Alphabet Dataset  
- Open-source ML community  

⭐ If you like this project, give it a star on GitHub!

