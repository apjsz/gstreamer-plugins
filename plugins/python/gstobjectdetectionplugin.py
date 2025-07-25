#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')

from gi.repository import Gst, GObject, GstBase, GstVideo
import numpy as np
import os
from pathlib import Path
import traceback
import json
import time


# Initializes GStreamer
Gst.init(None)

class GstObjectDetectionPlugin(GstBase.BaseTransform):
    """
    Object Detection with Metadata GStreamer Plugin
    """

    __gstmetadata__ = (
        "Object Detection with Metadata Plugin",
        "Filter/Video",
        "Runs object detection on video stream",
        "Your Name <your.email@example.com>"
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw,format={RGBA,RGB,BGR,RGBx,BGRx}")
        ),
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("video/x-raw,format={RGBA,RGB,BGR,RGBx,BGRx}")
        )
    )
    
    
    # Define GObject properties
    __gproperties__ = {
        "model-config": (GObject.TYPE_STRING,
                        "Model configuration file",
                        "Path to model configuration file (.prototxt)",
                        None,  # default value
                        GObject.ParamFlags.READWRITE),
        
        "model-weights": (GObject.TYPE_STRING,
                         "Model weights file",
                         "Path to model weights file (.caffemodel)",
                         None,
                         GObject.ParamFlags.READWRITE),
        
        "threshold": (GObject.TYPE_FLOAT,
                     "Confidence threshold",
                     "Minimum detection confidence (0.0 to 1.0)",
                     0.0, 1.0, 0.5,  # min, max, default
                     GObject.ParamFlags.READWRITE),

        "output-metadata": (GObject.TYPE_BOOLEAN,
                      "Output metadata",
                      "Enable metadata attachment to buffers",
                      True,
                      GObject.ParamFlags.READWRITE)
    }
    
    def __init__(self):
        super(GstObjectDetectionPlugin, self).__init__()
        self.video_info = None
        self.detector = None
        self.model_initialized = False

        # Initialize properties
        self.model_config = None
        self.model_weights = None
        self.threshold = 0.5
        self.output_metadata = True


   # Property handling
    def do_get_property(self, prop):
        if prop.name == 'model-config':
            return self.model_config
        elif prop.name == 'model-weights':
            return self.model_weights
        elif prop.name == 'threshold':
            return self.threshold
        elif prop.name == 'output-metadata':
            return self.output_metadata
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def do_set_property(self, prop, value):
        if prop.name == 'model-config':
            model_conf = value
            rel_config_path = Path(model_conf)
            self.model_config = rel_config_path.absolute()
            # print(f"Absolute path using pathlib.Path().absolute(): {self.model_config}")
            self._reset_detector()
        elif prop.name == 'model-weights':
            model_weight = value
            rel_weights_path = Path(model_weight)
            self.model_weights = rel_weights_path.absolute()
            # print(f"Absolute path using pathlib.Path().absolute(): {self.model_weights}")
            self._reset_detector()
        elif prop.name == 'threshold':
            self.threshold = float(value)
            if self.detector:
                self.detector.conf_threshold = self.threshold
        elif prop.name == 'output-metadata':
          self.output_metadata = bool(value)
        else:
            raise AttributeError(f'Unknown property {prop.name}')

    def _reset_detector(self):
        """Reset detector when model paths change"""
        # print(f"---> Model config: {self.model_config}")
        # print(f"---> Model weights: {self.model_weights}")

        self.detector = None
        self.model_initialized = False
        if self.model_config and self.model_weights:
            self._initialize_detector()

    def _initialize_detector(self):
        """Initialize detector with current properties"""
        if not self.model_config or not self.model_weights:
            Gst.warning("Model paths not set")
            return False
            
        try:
            from object_detector import ObjectDetector
            
            # Verify files exist
            if not os.path.exists(self.model_config):
                raise FileNotFoundError(f"Model config missing: {self.model_config}")
            if not os.path.exists(self.model_weights):
                raise FileNotFoundError(f"Model weights missing: {self.model_weights}")
            
            # Initialize detector
            self.detector = ObjectDetector(
                self.model_config, 
                self.model_weights,
                conf_threshold=self.threshold
            )
            self.model_initialized = True
            Gst.info(f"Object detector initialized with: {self.model_config}")
            return True
        except ImportError as e:
            Gst.error(f"Failed to import detector module: {e}")
        except Exception as e:
            Gst.error(f"Detector initialization failed: {str(e)}\n{traceback.format_exc()}")
        return False
    
    
    def do_set_caps(self, incaps, outcaps):
        try:
            Gst.info(f"Set caps: {incaps.to_string()}")
            self.video_info = GstVideo.VideoInfo.new_from_caps(incaps)
            # Initialize detector when caps are set
            if not self.model_initialized and self.model_config and self.model_weights:
                self._initialize_detector()
            return True
        except Exception as e:
            Gst.error(f"Set caps error: {str(e)}")
            return False
    

    def do_transform_ip(self, buf: Gst.Buffer) -> Gst.FlowReturn:
        if not self.video_info:
            Gst.error("Video info not available")
            return Gst.FlowReturn.NOT_LINKED
            
        if not self.model_initialized:
            Gst.error("Detector not initialized")
            return Gst.FlowReturn.NOT_LINKED

        try:
            width = self.video_info.width
            height = self.video_info.height
            Gst.info(f"Processing frame: {width}x{height}")
            
            
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:
                format = self.video_info.finfo.format
                if format == GstVideo.VideoFormat.RGBA:
                    frame_array = np.ndarray((height, width, 4), buffer=info.data, dtype=np.uint8)
                elif format == GstVideo.VideoFormat.RGB:
                    frame_array = np.ndarray((height, width, 3), buffer=info.data, dtype=np.uint8)
                elif format == GstVideo.VideoFormat.BGR:
                    frame_array = np.ndarray((height, width, 3), buffer=info.data, dtype=np.uint8)
                elif format in [GstVideo.VideoFormat.RGBx, GstVideo.VideoFormat.BGRx]:
                    frame_array = np.ndarray((height, width, 4), buffer=info.data, dtype=np.uint8)
                else:
                    Gst.warning(f"Unsupported format: {format}")
                    return Gst.FlowReturn.OK

                    
                # Process frame with detector
                processed_frame, detections = self.detector.process_frame(frame_array)
                
                # Ensure we're modifying the original buffer
                np.copyto(frame_array, processed_frame)

                # Attach metadata to buffer
                if self.output_metadata and detections:
                  self._attach_metadata(buf, detections)

            return Gst.FlowReturn.OK
        
        except Gst.MapError as e:
            Gst.error(f"Mapping error: {e}")
            return Gst.FlowReturn.ERROR
        except Exception as e:
            Gst.error(f"Detection error: {str(e)}\n{traceback.format_exc()}")
            return Gst.FlowReturn.ERROR


    def _attach_metadata(self, buf: Gst.Buffer, detections: list):
        try:
            # Get various timestamps
            system_time_ns = time.time_ns()
            clock = self.get_clock()
            clock_time = clock.get_time() if clock else 0

            # Create JSON metadata
            metadata = {
                "timestamps": {
                    "system": system_time_ns,
                    "pipeline": clock_time,
                    "pts": int(buf.pts),
                    "dts": int(buf.dts),
                    "duration": int(buf.duration)
                },
                "resolution": {
                    "width": int(self.video_info.width),
                    "height": int(self.video_info.height)
                },
                "detections": detections
            }
            json_str = json.dumps(metadata)

            # Create and post the message
            struct = Gst.Structure.new_empty("object-detection")
            struct.set_value("data", json_str)
            msg = Gst.Message.new_application(self, struct)
            self.get_parent().post_message(msg)
        
            """
            # Create a structure with metadata
            struct = Gst.Structure.new_empty("object-detection")
            struct.set_value("data", json_str)
            # struct.set_value("buffer_pts", int(buf.pts))

            msg = Gst.Message.new_application(self, struct)
            self.get_parent().post_message(msg)

            
            Gst.debug(f"Posted metadata for buffer {buf.pts}")
            """

        except Exception as e:
            Gst.error(f"Metadata attachment failed: {str(e)}")


# Register the plugin
GObject.type_register(GstObjectDetectionPlugin)
__gstelementfactory__ = (
    "objdetect",
    Gst.Rank.NONE,
    GstObjectDetectionPlugin
)



