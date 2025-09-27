# Point-to-Point Distance Measurement

This application provides real-time 3D distance measurement between two points using DepthAI. 
The backend processes video and depth streams to calculate precise 3D Euclidean distances, while the frontend provides an intuitive interface for point selection and measurement display.

The frontend, built using the @luxonis/depthai-viewer-common package, provides real-time video streams with interactive point selection. The backend uses DepthAI's depth estimation capabilities to calculate accurate 3D distances between selected points.

> **Note:** This example works on DepthAI devices with depth capabilities.

## Development Setup

### Prerequisites

- **Luxonis device** connected to your computer
- **Python+** for the backend
- **Node.js 16+** for the frontend
- Refer to the [documentation](https://docs.luxonis.com/software-v3/) to setup your device

### Quick Start

1. **Install Backend Dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install and Build Frontend:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

3. **Start Backend:**
   ```bash
   cd backend
   python src/main.py
   ```

4. **Start Frontend Preview:**
   ```bash
   cd frontend
   npm run preview
   ```

5. **Open the application:**
   - Frontend: `http://localhost:4173` (or the port shown in terminal)

### Backend Parameters

```
-d DEVICE, --device DEVICE
    Optional name, DeviceID or IP of the camera to connect to. (default: None)
-fps FPS_LIMIT, --fps-limit FPS_LIMIT
    FPS limit. (default: None)
-ip IP, --ip IP       IP address to serve the frontend on. (default: None)
-p PORT, --port PORT  Port to serve the frontend on. (default: None)
```

### How to Use

1. **Select Points:** Click on the video stream to select two points
2. **View Distance:** The 3D Euclidean distance will be displayed in real-time
3. **Clear Points:** Press **Space** or right-click to reset
4. **Toggle Tracking:** Use the tracking button to enable/disable point tracking
5. **Change Units:** Switch between metric (m) and imperial (ft) units
6. **Adjust Precision:** Select decimal places for distance display

## Production Deployment

### Standalone Mode (RVC4 only)

For production deployment on RVC4 devices, the app runs entirely on the device:

1. **Install oakctl:**
   ```bash
   # Follow installation instructions at:
   # https://docs.luxonis.com/software-v3/oak-apps/oakctl
   ```

2. **Build and Deploy:**
   ```bash
   # Build frontend for production
   cd frontend
   npm run build
   cd ..
   
   # Deploy to device
   oakctl connect <DEVICE_IP>
   oakctl app run .
   ```

3. **Access the app:**
   - Open `https://<OAK4_IP>:9000/` in your browser
   - The exact URL will be shown in the terminal output

## Features

- **Real-time 3D Distance Measurement:** Precise Euclidean distance calculation between two points
- **Interactive Point Selection:** Click-to-select interface with visual feedback
- **Unit Conversion:** Switch between metric (meters) and imperial (feet) units
- **Precision Control:** Adjustable decimal places for distance display
- **Tracking Modes:** Toggle between active tracking and static point display
- **Standard Deviation:** Display measurement uncertainty when available