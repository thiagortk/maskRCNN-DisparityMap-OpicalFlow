# maskRCNN with DisparityMap and OpicalFlow as post-processing
Using Disparity Map and Optical Flow info to extract features from the detected objects by maskRCNN.

**maskRCNN** example from: https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/

**Disparity Map**: We use here only the resulted Disparity Map image (see: https://github.com/thiagortk/stereoVision)

**Optical Flow**: It's in the code along with maskRCNN. We use the FarneBack algorithm available in OpenCV.
