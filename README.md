🎭 Face Filter Fun 🎩
An interactive and fun real-time face filter web application built with Streamlit, OpenCV, and MediaPipe. Apply virtual accessories like mustaches, hats, glasses, and wizard hats to your face live using your webcam — and download snapshots too!

📸 Features
🎥 Real-time webcam capture with face landmark detection (up to 4 faces)

🧔 Virtual filters: Mustache, Hat, Glasses, Wizard Hat, Grayscale

📷 Snapshot capture and download

🔢 Face count display

🎨 Stylish UI with gradient background

📱 Built-in QR code generator for sharing links

🛠️ Tech Stack
``` bash
Python

Streamlit

OpenCV

MediaPipe

NumPy

Pillow

qrcode
```

🚀 How to Run
Clone the repository

```
git clone https://github.com/usha1459/face-filter-app.git
cd face-filter-app
```

Install the dependencies
(Recommended: Use a virtual environment)
```
pip install -r requirements.txt
```

Place your overlay images

Ensure the following PNG images (with transparent background) are present in the root folder:
``` bash
mustache.png

hat.png

glasses.png

wizard_hat.png
```

Run the app

```bash
streamlit run app.py
```

📁 Project Structure
``` bash
face-filter-app/
│
├── app.py                   # Main Streamlit app
├── mustache.png             # Filter overlay
├── hat.png
├── glasses.png
├── wizard_hat.png
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

📦 requirements.txt (example content)

streamlit
opencv-python
mediapipe
numpy
qrcode
Pillow


🙋‍♀️ Made With ❤️ by Prathyusha
If you like it, 🌟 star the repo or connect with me!
