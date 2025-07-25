 
import numpy as np
import cv2
import os

class ObjectDetector:
    def __init__(self, model_config, model_weights, conf_threshold=0.5):
        """
        Initialize object detector with separate config and weights
        Args:
            model_config: Path to model configuration file (prototxt)
            model_weights: Path to model weights file (caffemodel)
            conf_threshold: Minimum confidence threshold
        """
        self.conf_threshold = conf_threshold

        # print(f"model config path: {model_config}")
        # print(f"model weights path: {model_weights} ")
        
        # Verify files exist
        if not os.path.exists(model_config):
            raise FileNotFoundError(f"Model config missing: {model_config}")
        if not os.path.exists(model_weights):
            raise FileNotFoundError(f"Model weights missing: {model_weights}")

        # Load the model
        self.net = cv2.dnn.readNetFromCaffe(model_config, model_weights)
        
        # Get output layer names - compatible with all OpenCV versions
        layer_names = self.net.getLayerNames()
        try:
            # Newer OpenCV versions (4.x)
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            # Older OpenCV versions (3.x)
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        """
        # COCO classes (91 items)
        self.classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        """
        # COCO classes (20 items)
        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"
        ]
        
        """
        # Debug output
        print(f"Loaded model: {model_config}")
        print(f"Output layers: {self.output_layers}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Number of classes: {len(self.classes)}")
        """
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process frame with object detection
        Args:
            frame: Input RGBA frame (HxWx4)
        Returns:
            Frame with detections drawn (RGBA)
        """
        # Convert to BGR for OpenCV DNN
        """
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        height, width = bgr.shape[:2]
        """

        # Handle different input formats
        input_channels = frame.shape[2]
        if input_channels == 4:
            # RGBA input - convert to BGR temporarily
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif input_channels == 3:
            # RGB/BGR input
            bgr = frame.copy()
        else:
            raise ValueError(f"Unsupported channel count: {input_channels}")

        height, width = bgr.shape[:2]
        
        # Create blob and run detection
        blob = cv2.dnn.blobFromImage(bgr, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward(self.output_layers)
        
        # Get detections - compatible with different output formats
        detections_list = []
        if len(detections) == 1:
            # Single output layer
            detections = detections[0]
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                class_id = int(detections[0, 0, i, 1])
                print(f"Class Id: {class_id}")
                # Bounding box coordinates are normalized [0,1]
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding box stays within image
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(width, endX)
                endY = min(height, endY)

                # Store detection metadata
                detections_list.append({
                    "class_id": class_id,
                    "class_name": self.classes[class_id],
                    "confidence": float(confidence),
                    "x": float(startX),
                    "y": float(startY),
                    "width": float(endX - startX),
                    "height": float(endY - startY)
                })
                
                # Draw bounding box
                label = f"{self.classes[class_id]}: {confidence:.2f}"
                color = (0, 255, 0)
                if input_channels == 4:
                  color = (0, 255, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                # Draw label background
                y = startY - 15 if startY - 15 > 15 else startY + 15
                # Estimate text size for background
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (startX, y - text_height - 5), 
                             (startX + text_width, y), 
                             color, cv2.FILLED)
                
                # Draw label text
                text_color = (0, 0, 0) if input_channels == 3 else (0, 0, 0, 255)
                cv2.putText(frame, label, (startX, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        return frame, detections_list
