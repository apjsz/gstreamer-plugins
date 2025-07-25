#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import json

Gst.init(None)

# Pipeline as a single string
pipeline_description = (
    "v4l2src device=/dev/video0 ! "
    "videoconvert ! video/x-raw,format=RGBA ! "
    "objdetect name=detector model_config=models/MNSSD.prototxt model_weights=models/MNSSD.caffemodel output-metadata=True ! "
    "videoconvert ! autovideosink sync=false"
)

pipeline = Gst.parse_launch(pipeline_description)

# Get the bus and add a watch
bus = pipeline.get_bus()
bus.add_signal_watch()

def on_message(bus, message, loop):
    msg_type = message.type

    if msg_type == Gst.MessageType.EOS:
        print("End-Of-Stream reached.")
        loop.quit()

    elif msg_type == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err.message}")
        print(f"Debug info: {debug}")
        loop.quit()

    elif msg_type == Gst.MessageType.APPLICATION:
        struct = message.get_structure()
        if struct and struct.get_name() == "object-detection":
            try:
                json_str = struct.get_string("data")
                if json_str:
                    detections = json.loads(json_str)
                    print(f"[Metadata] Detections:\n{json.dumps(detections, indent=2)}")
            except Exception as e:
                print(f"[Error parsing metadata]: {e}")

    return True

loop = GLib.MainLoop()
bus.connect("message", on_message, loop)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)
try:
    print("Pipeline running. Press Ctrl+C to stop.")
    loop.run()
except KeyboardInterrupt:
    print("Stopping pipeline...")

# Clean up
pipeline.set_state(Gst.State.NULL)
