# Face Enrolment System - Hybrid MediaPipe TFLite

## 🎯 Overview

**Hybrid face enrollment system** that uses MediaPipe TensorFlow Lite models for high-accuracy face detection and landmarks, with OpenCV as automatic fallback. Completely offline processing with no internet dependency.

## 📁 Project Structure

```
mini-project/
├── enrolment.py              # Main hybrid system
├── mediapipe_models/         # TFLite models (required)
│   ├── mediapipe_face_detection_short_range.tflite
│   └── mediapipe_face_landmark.tflite
├── database/
│   ├── enrolment.pkl        # Main database
│   └── enrolment_backup.pkl # Automatic backup
└── requirements.txt
```

## ✨ Features

- **🎯 Hybrid Processing**: MediaPipe TFLite → OpenCV fallback
- **📍 468 Landmarks**: Enhanced feature extraction
- **🔒 Offline**: No internet required
- **💾 Auto-Backup**: Database backup on each save
- **⚡ XNNPACK**: Hardware acceleration

## 🎮 Usage

### 1. Prepare Images

```
photos/
├── person1/
│   ├── photo1.jpg
│   ├── photo2.png
│   └── photo3.jpeg
└── person2/
    ├── img1.jpg
    └── img2.jpg
```

### 2. Run System

```bash
python enrolment.py
```

### 3. Menu Options

1. **👤 Add New Registration** - Process folder of images
2. **💾 Save Database & Exit** - Save with backup
3. **📊 View Registrations** - Show enrolled persons
4. **❌ Exit without saving** - Discard changes

## 🔧 Technical Details

### Processing Pipeline

1. **Primary**: MediaPipe TFLite face detection (128×128 input)
2. **Landmarks**: 468 facial points (192×192 input)
3. **Fallback**: OpenCV Haar cascades
4. **Features**: 512-dimensional embeddings

### Database Format

```python
{
    'metadata': {
        'total_persons': 2,
        'processing_mode': 'direct_tflite'
    },
    'persons': {
        'john_doe': {
            'embeddings': [...],
            'image_count': 5,
            'processing_info': {
                'method': 'tflite',
                'landmarks_extracted': True
            }
        }
    }
}
```

## � Troubleshooting

**No faces detected?**

- Ensure face size >80×80 pixels
- Good lighting and contrast
- Front-facing poses work best

**TFLite issues?**

- System auto-falls back to OpenCV
- Check models exist in `mediapipe_models/`

**🎯 Ready to use! Your hybrid face enrollment system with MediaPipe TFLite + OpenCV fallback.**
