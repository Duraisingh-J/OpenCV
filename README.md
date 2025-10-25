# 🎥 Real-Time AI Vision Assistant

<div align="center">

![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-blue.svg)
![Python](https://img.shields.io/badge/Python-3.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**A powerful, multi-functional computer vision application combining real-time face detection, object tracking, motion sensing, and interactive drawing capabilities.**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 🌟 Features

<table>
<tr>
<td width="50%">

### 🎯 Detection & Recognition
- **Face Detection**: Real-time frontal face detection with eye tracking
- **Motion Detection**: Advanced background subtraction for movement sensing
- **Edge Detection**: Canny edge detection with corner identification
- **Performance Monitoring**: Live FPS counter and optimization

</td>
<td width="50%">

### 🎮 Interactive Features
- **Object Tracking**: CSRT-based robust object tracking
- **Distance Measurement**: Click-to-measure pixel distance tool
- **Air Drawing**: Draw in 3D space using colored objects
- **Screenshot Capture**: Save any frame with one keypress

</td>
</tr>
</table>

---

## 🎬 Demo

### Mode Overview

| Mode | Description | Use Case |
|------|-------------|----------|
| 🔍 **Detection** | Face, eye, and motion detection | Security, monitoring, attendance |
| 🎯 **Tracking** | Follow any selected object | Sports analysis, surveillance |
| 📏 **Measurement** | Measure distances between points | Spatial analysis, calibration |
| 🎨 **Drawing** | Air drawing with color tracking | Interactive presentations, fun |
| 🔲 **Edges** | Edge and corner visualization | Scene understanding, alignment |

---

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam or camera device
- Windows, macOS, or Linux

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/vision-assistant.git
cd vision-assistant

# Install dependencies
pip install -r requirements.txt

# Run the application
python vision_assistant.py
```

### Manual Installation

```bash
# Install required packages
pip install opencv-python>=4.8.0
pip install opencv-contrib-python>=4.8.0
pip install numpy>=1.21.0
```

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Starting the Application

```bash
python vision_assistant.py
```

### Keyboard Controls

| Key | Action | Description |
|-----|--------|-------------|
| `1` | Detection Mode | Face, eye, and motion detection |
| `2` | Tracking Mode | Click and drag to select object |
| `3` | Measurement Mode | Click two points to measure |
| `4` | Drawing Mode | Use green object to draw |
| `5` | Edge Mode | View edges and corners |
| `C` | Clear/Reset | Clear drawings or reset tracking |
| `S` | Screenshot | Save current frame |
| `Q` | Quit | Exit application |

### Mode-Specific Instructions

#### 🔍 Detection Mode (Key: 1)
Simply face the camera. The system will automatically:
- Detect faces (green boxes)
- Identify eyes (blue boxes)
- Track motion (red boxes)

#### 🎯 Tracking Mode (Key: 2)
1. Press `2` to enter tracking mode
2. Click and drag to select the object
3. Release to start tracking
4. Press `C` to reset and select new object

#### 📏 Measurement Mode (Key: 3)
1. Press `3` to enter measurement mode
2. Click first point
3. Click second point
4. Distance displays automatically
5. Press `C` to clear and start new measurement

#### 🎨 Drawing Mode (Key: 4)
1. Press `4` to enter drawing mode
2. Hold a green object (marker, paper, toy)
3. Move to draw in the air
4. Press `C` to clear canvas

#### 🔲 Edge Mode (Key: 5)
Automatically displays:
- Yellow dots: Detected corners
- Colored overlay: Detected edges

---

## 📚 Documentation

### Architecture

```
VisionAssistant/
├── Face & Eye Detection     → Haar Cascade Classifiers
├── Motion Detection         → Background Subtraction (MOG2)
├── Object Tracking          → CSRT Tracker
├── Edge Detection          → Canny Algorithm
├── Corner Detection        → Shi-Tomasi Algorithm
└── Color Tracking          → HSV Color Space
```

### Configuration

#### Camera Settings

```python
# Change camera index (default: 0)
self.cap = cv2.VideoCapture(1)  # Try different indices

# Adjust resolution
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

#### Color Tracking Calibration

For drawing mode, adjust HSV values based on your lighting:

```python
# In air_drawing method (around line 115)
lower_green = np.array([35, 50, 50])   # Lower bound
upper_green = np.array([85, 255, 255]) # Upper bound
```

**HSV Color Ranges:**
- Red: [0, 50, 50] to [10, 255, 255]
- Blue: [100, 50, 50] to [130, 255, 255]
- Yellow: [20, 50, 50] to [30, 255, 255]

#### Performance Optimization

```python
# Reduce frame size for better FPS
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Adjust detection sensitivity
faces = self.face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.3,  # Increase for speed
    minNeighbors=5    # Decrease for more detections
)
```

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><b>Camera Not Opening</b></summary>

```python
# Try different camera indices
self.cap = cv2.VideoCapture(1)  # or 2, 3, etc.

# Check available cameras
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()
```
</details>

<details>
<summary><b>OpenCV GUI Error</b></summary>

```bash
# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```
</details>

<details>
<summary><b>Slow Performance</b></summary>

- Reduce camera resolution
- Close other applications
- Update graphics drivers
- Use GPU acceleration (install opencv-python with CUDA)
</details>

<details>
<summary><b>Drawing Mode Not Working</b></summary>

- Ensure good lighting
- Use a bright green object
- Adjust HSV color range in code
- Test with different colored objects
</details>

---

## 🎯 Use Cases

### Professional Applications
- 🏢 **Security Systems**: Motion detection and face recognition
- 👨‍🏫 **Education**: Interactive computer vision demonstrations
- 🏥 **Healthcare**: Patient monitoring and movement tracking
- 🏭 **Industrial**: Quality control and measurement
- 🎬 **Media**: Video analysis and tracking

### Personal Projects
- 🎮 **Gaming**: Motion-controlled interfaces
- 🎨 **Art**: Interactive installations
- 📸 **Photography**: Object tracking for cameras
- 🏠 **Smart Home**: Automated surveillance
- 🎓 **Learning**: OpenCV tutorial and experimentation

---

## 🛠️ Tech Stack

- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **Python**: Core programming language

### Key Algorithms
- Haar Cascade Classifiers (Face/Eye Detection)
- MOG2 Background Subtractor (Motion Detection)
- CSRT Tracker (Object Tracking)
- Canny Edge Detection
- Shi-Tomasi Corner Detection
- HSV Color Space Analysis

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Average FPS | 30-60 (depends on hardware) |
| Face Detection | Real-time, multiple faces |
| Tracking Accuracy | >90% under good conditions |
| Motion Sensitivity | Configurable (500+ pixel area) |
| Resolution | Up to 1920x1080 (configurable) |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🔨 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🎉 Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/vision-assistant.git
cd vision-assistant

# Create branch
git checkout -b feature/your-feature

# Make changes and test
python vision_assistant.py

# Submit PR
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- OpenCV Community for excellent documentation
- Python Software Foundation
- All contributors and testers

---

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/yourserver)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/vision-assistant/issues)

---

## 🗺️ Roadmap

- [ ] Hand gesture recognition
- [ ] QR code scanning
- [ ] Facial landmark detection
- [ ] Multi-object tracking
- [ ] Video recording capability
- [ ] Mobile app version
- [ ] GPU acceleration support
- [ ] Custom model training interface

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ and OpenCV

[Report Bug](https://github.com/yourusername/vision-assistant/issues) · [Request Feature](https://github.com/yourusername/vision-assistant/issues) · [Documentation](https://github.com/yourusername/vision-assistant/wiki)

</div>