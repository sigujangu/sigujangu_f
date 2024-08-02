import streamlit as st
import cv2
import numpy as np
import io
import utils

temporary_location = None

def play_video(video_source, model):
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    detected_message = st.empty()

    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            visualize_image, detected = utils.predict_image(frame, conf_threshold)
            st_frame.image(visualize_image, channels="BGR")
            
            if detected:
                detected_message.warning("Lionfish detected!")
            else:
                detected_message.warning("No Lionfish detected")

        else:
            camera.release()
            break

st.set_page_config(
    page_title="AquaGuard",
    page_icon=":fish:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Lionfish Detector :fish:")

st.sidebar.header("Type")
source_radio = st.sidebar.radio("Select Source", ["VIDEO"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20)) / 100

if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "mov", "avi", "mkv"])

    if input is not None:
        temporary_location = "uploaded_video.mp4"

        g = io.BytesIO(input.read())  ## BytesIO Object
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file

        st.video(input)

if temporary_location is not None:
    play_video(temporary_location, utils.compiled_model_face)
