import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
from PIL import Image
import cv2 as cv2
from sudokudns import *


def video_frame_callback(frame):
    img = frame.to_ndarray(format="rgb24")
    im = Image.fromarray(img)
    im.save("temp.jpeg")

    img = frame.to_ndarray(format="bgr24")

    cv_img = cv2.imread("./temp.jpeg")
    output = solveSudoku(cv_img)

    return av.VideoFrame.from_ndarray(output, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
