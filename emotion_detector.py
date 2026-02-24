# emotion_detector.py
import cv2
from fer import FER
import time

class EmotionDetector:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.last_detection_time = 0
        self.detection_interval = 1  # seconds
        self.current_emotion = "neutral"
        self.emotion_history = []
        
    def detect_emotion_from_frame(self, frame):
        """Detect emotion from frame, but only if enough time has passed"""
        current_time = time.time()
        
        # Only detect emotion every 10 seconds
        if current_time - self.last_detection_time >= self.detection_interval:
            try:
                result = self.detector.top_emotion(frame)
                if result:
                    emotion, score = result
                    if score > 0.5:  # Only accept if confidence is high enough
                        self.current_emotion = emotion
                        self.emotion_history.append((emotion, current_time))
                        # Keep only last 10 emotions
                        if len(self.emotion_history) > 10:
                            self.emotion_history.pop(0)
                
                self.last_detection_time = current_time
            except Exception as e:
                print(f"Emotion detection error: {e}")
        
        return self.current_emotion
    
    def get_emotion_streak(self):
        """Check how long the same emotion has persisted"""
        if len(self.emotion_history) < 2:
            return 1
        
        current = self.emotion_history[-1][0]
        streak = 1
        
        for emotion, _ in reversed(self.emotion_history[:-1]):
            if emotion == current:
                streak += 1
            else:
                break
        
        return streak
    
    def get_dominant_emotion(self, window_seconds=30):
        """Get the most frequent emotion in the last window_seconds"""
        cutoff_time = time.time() - window_seconds
        recent = [e for e, t in self.emotion_history if t > cutoff_time]
        
        if not recent:
            return self.current_emotion
        
        from collections import Counter
        counts = Counter(recent)
        return counts.most_common(1)[0][0]

def start_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None
    
    # Initialize emotion detector
    detector = EmotionDetector()
    
    return cap, detector