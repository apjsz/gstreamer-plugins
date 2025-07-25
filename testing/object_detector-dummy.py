# Dummy object_detector.py
# Genera un patrón sobre una entrada para ensayar la negociación de los caps.

import numpy as np
import time

class DummyObjectDetector:
    def __init__(self, *args, **kwargs):
        self.process_time = 0.01  # Simulate 10ms processing time
        print("Dummy detector initialized successfully!")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Simulates object detection by:
        1. Adding a timestamp overlay
        2. Drawing a moving test pattern
        3. Adding a processing delay
        """
        # Add processing delay
        time.sleep(self.process_time)
        
        height, width, _ = frame.shape
        
        # Draw timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        text = f"Dummy Detector | {timestamp}"
        for i, char in enumerate(text):
            y_pos = 20 + (i % 3) * 10
            x_pos = 10 + i * 10
            frame[y_pos:y_pos+8, x_pos:x_pos+8, :3] = [255, 0, 0]  # Red pixels
        
        # Draw moving test pattern
        t = time.time()
        offset = int(t * 10) % 100
        frame[offset:offset+5, :, 0] = 255  # Red bar
        frame[:, offset:offset+5, 1] = 255   # Green bar
        
        return frame