
import cv2
import streamlit as st
import time
from emotion_detector import EmotionDetector
from datetime import datetime

def run_live_camera_window():
    """ Runs a cv2.imshow loop for live emotion detection """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open camera.")
        return

    detector = EmotionDetector()
    st.toast("Camera started! Press 'q' in the window to stop.")
    
    # We need to access session state, but we can't update it safely from here if running in a thread
    # But since this is running in the main script flow (blocking), we can try to update it
    # However, Streamlit won't reflect changes until rerun.
    # So this loop is mainly for the *experience* of the window.
    
    last_check_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect emotion
        current_time = time.time()
        emotion = detector.detect_emotion_from_frame(frame)
        
        # Draw on frame
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Task Optimizer - Live Monitor', frame)
        
        # Check if we should update the global state (e.g. every 2 seconds)
        if current_time - last_check_time > 2.0:
            if emotion != st.session_state.current_emotion:
                st.session_state.current_emotion = emotion
                st.session_state.emotion_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "emotion": emotion
                })
                # We can't easily trigger the UI update while stuck in this loop
                # But we can update the state so when they close, it's there.
            last_check_time = current_time
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    st.rerun()
