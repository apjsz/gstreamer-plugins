# gstreamer-plugins
Plugins for GStreamer

Plugin template and some plugins for video processing using machine learning and  deep learning model.

How to use the plugin and a sample path:

export GST_PLUGIN_PATH=<your plugin path>

Object detector plugin. The files are:

gstobjectdetectionplugin.py (generic)
object_detector.py (vith the model)

gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! video/x-raw,format=RGBA ! objdetect name=detector model_config=models/MNSSD.prototxt model_weights=models/MNSSD.caffemodel output-metadata=True ! videoconvert ! xvimagesink sync=false

Or with a test file that shows the metadata.

python run_with_metadata_listener.py
 
