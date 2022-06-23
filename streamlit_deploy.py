import streamlit as st
import model
from streamlit_webrtc import webrtc_streamer
import av


def video_frame_callback(frame):
    img = blur_face(frame)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example")