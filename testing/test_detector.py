# test_detector.py
import cv2
import traceback
from object_detector import ObjectDetector

try:
    # Initialize detector
    detector = ObjectDetector("models/MobileNetSSD_deploy.prototxt", 
                              "models/MobileNetSSD_deploy.caffemodel")
    
    # Process test image
    img = cv2.imread("test.jpg", cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("Test image not found")
    
    if img.shape[2] == 3:  # Convert to RGBA if needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        
    processed = detector.process_frame(img)
    
    # Save result
    cv2.imwrite("output.jpg", cv2.cvtColor(processed, cv2.COLOR_RGBA2BGR))
    print("Processing complete! Output saved as output.jpg")
    
except Exception as e:
    print(f"Error: {str(e)}")
    print(traceback.format_exc())