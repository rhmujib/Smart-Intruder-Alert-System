# Smart Intruder Alert System 🔒

A **Smart Intruder Alert System** is a real-time surveillance solution designed to detect unauthorized intrusions, trigger alarms, and record video footage. This system leverages advanced computer vision and machine learning technologies to ensure the safety of your premises. It also provides a web-based interface for monitoring, controlling, and analyzing detection data.

---

## Features 🚀

- **Real-Time Intruder Detection**: Uses a pre-trained MobileNet SSD model to detect intruders in live video streams with high accuracy.
- **Alarm System**: Triggers an alarm when an intruder is detected to alert the user immediately.
- **Video Recording**: Automatically records video footage of intrusions for future reference.
- **Interactive Web Interface**: A user-friendly Flask-based web interface for monitoring and controlling the system.
- **Historical Data Visualization**: Displays historical intruder data using interactive graphs.
- **Dashboard Analytics**: Provides real-time statistics, including intruder count, authorized person count, and detection accuracy.
- **Customizable Settings**: Allows users to start, stop, or terminate the detection process with ease.

---

## How It Works 🛠️

1. **Live Video Feed**: The system captures a live video feed from a connected camera.
2. **Intruder Detection**: The video frames are processed using a TensorFlow-based MobileNet SSD model to detect intruders in real-time.
3. **Alarm Trigger**: If an intruder is detected, an alarm is triggered to alert the user.
4. **Video Recording**: The system records video footage of the intrusion and saves it for future reference.
5. **Web Interface**: Users can monitor the live video feed, view real-time statistics, and analyze historical data through an interactive web interface.

---

## Project Structure 📂
ASIAS/ ├── main.py # Main application file ├── templates/ │ └── index.html # HTML file for the web interface ├── static/ │ ├── style.css # CSS file for styling the web interface │ └── recordings/ # Folder for storing recorded videos ├── models/ │ └── mobilenetssd/ # Pre-trained MobileNet SSD model files ├── requirements.txt # Python dependencies ├── README.md # Project documentation └── history_log.json # JSON file for storing historical detection data


---

## Installation and Setup 🖥️

Follow these steps to set up and run the project:

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- A webcam or external camera

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Smart-Intruder-Alert-System.git
   cd Smart-Intruder-Alert-System

2. Install the required dependencies:
pip install -r requirements.txt

3. Run the application:
python main.py

4. Open your browser and navigate to:
http://127.0.0.1:5000/

Usage Instructions 📖
Start Detection: Click the "Start Detection" button on the web interface to begin monitoring.
Stop Detection: Click the "Stop Detection" button to pause the monitoring process.
View Real-Time Data: Monitor intruder statistics and live video feed on the dashboard.
Analyze Historical Data: View historical intruder and authorized person data in the "Historical Graph" section.
Reset Data: Use the "Reset Graph" button to clear historical data.
Technologies Used 🛠️
Backend: Flask, Flask-SocketIO
Frontend: HTML, CSS, JavaScript, Chart.js, SweetAlert2
Machine Learning: TensorFlow, MobileNet SSD
Computer Vision: OpenCV
Audio: Pygame (for alarm system)


Key Features in Detail 📊
Real-Time Intruder Detection
The system uses the MobileNet SSD model to detect intruders in live video streams. It processes each frame in real-time and identifies objects classified as "person."

Alarm System
When an intruder is detected, the system triggers an alarm sound to alert the user. The alarm continues until the intruder is no longer detected.

Video Recording
The system records video footage of intrusions and saves it in the recordings folder. Each recording is timestamped for easy identification.

Interactive Web Interface
The web interface provides a clean and minimalist design for monitoring and controlling the system. Users can view live video feeds, real-time statistics, and historical data.

Historical Data Visualization
The system stores detection data in a JSON file (history_log.json) and visualizes it using interactive graphs. Users can analyze trends and patterns in intruder activity.

Future Scope 🔮

IoT Integration: Connect the system with IoT devices for enhanced automation.
Multi-Camera Support: Add support for monitoring multiple camera feeds simultaneously.
Facial Recognition: Implement advanced facial recognition to identify authorized personnel.
Cloud Storage: Store historical data and video recordings in the cloud for remote access.
Mobile App: Develop a mobile app for remote monitoring and control.

Contributing 🤝
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch:
git checkout -b feature-name

3. Commit your changes:
git commit -m "Add feature-name"

4. Push to the branch:
git push origin feature-name

5. Open a pull request.

License 📜
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments 🙌
TensorFlow: For providing the pre-trained MobileNet SSD model.
Flask: For the lightweight and flexible web framework.
Chart.js: For creating interactive graphs.
SweetAlert2: For beautiful alert popups.
Contact 📧
For any inquiries or feedback, please contact:

Email: mujeebah789@gmail.com
GitHub: rhmujib
Thank you for using the Smart Intruder Alert System! 😊

