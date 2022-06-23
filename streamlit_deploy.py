import streamlit as st
import model
from streamlit_webrtc import webrtc_streamer
import av


class VideoProcessor:
    def video_frame_callback(self, frame):
        img = blur_face(frame)

        return av.VideoFrame.from_ndarray(img, format="bgr")


webrtc_streamer(key="example", video_processor_factory=VideoProcessor)