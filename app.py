import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import qrcode
from PIL import Image
from datetime import datetime

# Streamlit config
st.set_page_config(page_title="Face Filter Fun 🌭", page_icon="🎩", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #a8edea, #fed6e3);
            color: #000000;
        }
        h1, h2, h3, h4, h5, h6, p, div, span, .element-container {
            color: #000000 !important;
        }
        .stButton>button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #000000;
            border-radius: 10px;
            padding: 8px 20px;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load RGBA overlay images
mustache = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)
wizard_hat = cv2.imread("wizard_hat.png", cv2.IMREAD_UNCHANGED)

# Check if all images loaded
for name, img in [("mustache.png", mustache), ("hat.png", hat), ("glasses.png", glasses), ("wizard_hat.png", wizard_hat)]:
    if img is None or img.shape[2] != 4:
        st.error(f"'{name}' is missing or not a valid RGBA image.")
        st.stop()

# Face mesh setup
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=4)

# Overlay image utility
def overlay_rgba(bg, overlay, x, y, w, h):
    if w <= 0 or h <= 0:
        return bg

    overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
    b, g, r, a = cv2.split(overlay_resized)
    alpha = a / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])
    h_bg, w_bg = bg.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(x + w, w_bg), min(y + h, h_bg)
    x_overlay_start = x0 - x
    x_overlay_end = x1 - x
    y_overlay_start = y0 - y
    y_overlay_end = y1 - y

    if x_overlay_end <= x_overlay_start or y_overlay_end <= y_overlay_start:
        return bg

    overlay_slice = (slice(y_overlay_start, y_overlay_end), slice(x_overlay_start, x_overlay_end))
    bg_roi = (slice(y0, y1), slice(x0, x1))

    fg = cv2.merge([b, g, r])[overlay_slice]
    alpha_roi = alpha[overlay_slice]
    bg_part = bg[bg_roi]

    try:
        blended = cv2.convertScaleAbs(fg * alpha_roi + bg_part * (1 - alpha_roi))
        bg[bg_roi] = blended
    except:
        pass

    return bg

# App title
st.markdown("<h1 style='text-align:center;'>🌭 Real-Time Face Filter App 🎩</h1>", unsafe_allow_html=True)

# Sidebar: QR Code Generator
st.sidebar.markdown("### 🔗 QR Code Generator")
qr_input = st.sidebar.text_input("Enter link to generate QR")
if qr_input:
    qr = qrcode.make(qr_input)
    st.sidebar.image(qr, caption="Scan this QR", use_column_width=False)

# Layout
cols = st.columns([1, 3])

with cols[0]:
    st.markdown("### 🎨 Choose a Filter")
    filter_option = st.radio("Select a filter", [
        "Mustache",
        "Hat",
        "Glasses",
        "Wizard Hat",
        "Mustache & Hat",
        "Grayscale Filter"
    ])
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. **Choose a filter** from the left panel.  
    2. **Look into the camera** to see the filter in real-time.  
    3. **Click the 'Capture Snapshot'** button to save the image.  
    4. **Press 'Q' key** to quit the webcam preview.  
    """)

frame_placeholder = cols[1].empty()
capture_btn = cols[1].button("📸 Capture Snapshot")
captured = False
cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        face_count = 0
        if results.multi_face_landmarks:
            faces = results.multi_face_landmarks[:4]  # Only first 4 faces
            face_count = len(faces)

            for face in faces:
                landmarks = face.landmark

                def to_px(idx):
                    try:
                        pt = landmarks[idx]
                        return int(pt.x * w), int(pt.y * h)
                    except:
                        return None

                # Landmark points
                p1 = to_px(13)   # Nose
                p2 = to_px(14)   # Upper lip
                pl = to_px(127)  # Left face
                pr = to_px(356)  # Right face
                pf = to_px(10)   # Forehead
                le = to_px(33)   # Left eye
                re = to_px(263)  # Right eye
                fc = to_px(151)  # Top of head

                if None in (p1, p2, pl, pr, pf, le, re, fc):
                    continue

                x1, y1 = p1
                x2, y2 = p2
                x_l, _ = pl
                x_r, _ = pr
                x_f, y_f = pf
                x_le, y_le = le
                x_re, y_re = re
                x_fc, y_fc = fc

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                face_width = x_r - x_l

                # Filters
                if filter_option in ["Mustache", "Mustache & Hat"]:
                    m_w, m_h = face_width, int(face_width * 0.3)
                    m_x, m_y = cx - m_w // 2, cy - int(m_h * 0.75)
                    frame = overlay_rgba(frame, mustache, m_x, m_y, m_w, m_h)

                if filter_option in ["Hat", "Mustache & Hat"]:
                    h_w = int(face_width * 1.6)
                    h_h = int(h_w * 0.9)
                    h_x = x_f - h_w // 2
                    h_y = y_f - int(h_h * 0.8)
                    frame = overlay_rgba(frame, hat, h_x, h_y, h_w, h_h)

                if filter_option == "Glasses":
                    g_w = int(face_width * 1.1)
                    g_h = int(g_w * 0.4)
                    g_x = (x_le + x_re) // 2 - g_w // 2
                    g_y = (y_le + y_re) // 2 - g_h // 2
                    frame = overlay_rgba(frame, glasses, g_x, g_y, g_w, g_h)

                if filter_option == "Wizard Hat":
                    w_w = int(face_width * 1.8)
                    w_h = int(w_w * 1.2)
                    w_x = x_fc - w_w // 2
                    w_y = y_fc - int(w_h * 1.5)
                    frame = overlay_rgba(frame, wizard_hat, w_x, w_y, w_w, w_h)

        # Grayscale
        if filter_option == "Grayscale Filter":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Face count
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Detected Faces: {face_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        bordered_frame = cv2.copyMakeBorder(display_frame, 10, 10, 10, 10,
                                            cv2.BORDER_CONSTANT, value=[150, 200, 255])
        frame_placeholder.image(cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if capture_btn and not captured:
            clean_frame = frame.copy()
            _, buffer = cv2.imencode(".jpg", clean_frame)
            st.download_button("📅 Download Snapshot", buffer.tobytes(),
                               file_name="filtered_snapshot.jpg",
                               mime="image/jpeg",
                               key=f"download_btn_{datetime.now().timestamp()}")
            captured = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

st.markdown("---")
st.markdown("### 👩‍💻 Made by Prathyusha")
