import av
import time
import threading
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from utils.keypoint import DetectHandKeypoint
from utils.classifier import Classifier


# Passing the values between inside and outside the callback
lock = threading.Lock()


class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        # Init detection and classification models
        self.detector = DetectHandKeypoint()
        self.classifier = Classifier('models/classification.pt')
        self.img_pred = None
        
        # Load scaler parameters
        scaler_minmax_path = 'models/scaler_minmax.npy'
        self.min, self.max = np.load(scaler_minmax_path)
        
        self.prediction = '❌'
        self.counter = 0
        self.start = time.time()
        self.fps = 0

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format='bgr24')
        self.img_pred = img.copy()
        
        with lock:
            self.counter += 1
            self.fps = self.counter / (time.time() - self.start)
            
            results = self.detector(img)
            
            if results.multi_hand_landmarks:
                keypoints = self.detector.get_keypoint_list(results.multi_hand_landmarks[0].landmark)
                scaled_keypoints = scaled_keypoints = np.subtract(keypoints, self.min) / np.subtract(self.max, self.min)
                self.prediction = self.classifier(scaled_keypoints)
                self.img_pred = self.detector.plot(self.img_pred, results.multi_hand_landmarks[0])

            else:
                self.prediction = '❌'
                self.img_pred = img
                
        return av.VideoFrame.from_ndarray(img, format='bgr24')


def main():
    # Main content
    st.set_page_config(layout='wide', page_title='Gesture Recognition Based on Keypoints', page_icon='🤘')
    col1, col2 = st.columns(2)
    
    with col1:
        st.header('Gesture Recognition Based on Keypoints')
        
        ctx = webrtc_streamer(
            key='stream-detection',
            video_processor_factory=VideoProcessor,
            media_stream_constraints={'video': {'width': 640}, 'audio': False},
            # STUN/TURN server is required
            rtc_configuration={'iceServers': [{"urls": ["stun:stun.l.google.com:19302"]}]},
        )
        
    with col2:
        placeholder = st.empty()

        while ctx.state.playing:
            with lock:
                img_pred = ctx.video_processor.img_pred
                result = ctx.video_processor.prediction
                fps = ctx.video_processor.fps
                
                with placeholder.container():
                    st.header(f'Result: {result} | FPS: {round(fps, 2)}')
                    if img_pred is not None:
                        st.image(img_pred, use_column_width='always', channels='BGR')
        
            time.sleep(0.001)

if __name__ == '__main__':
    main()
