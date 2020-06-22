# maskRCNN with DisparityMap and OpicalFlow as post-processing
Using Disparity Map and Optical Flow info to extract features from the detected objects by maskRCNN.

**maskRCNN** example edited from: https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/

**Download and extract the model files**: <br/>
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz <br/>
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

**Caution**: It has absolute path in the code for the rectified images that should be used for the construction of the Disparity Map. You will need to change the absolute paths or change to use argv.

**Disparity Map**: First of all you need to build the disparity map images. To do this, we use the Stereo Vision code in filteredDisparityMap.cpp

To compile, type in your terminal: g++ filteredDisparityMap.cpp -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_calib3d -lopencv_features2d -lopencv_ximgproc -o veRun

If you need calibrate your stereo setup or rectify your images see the code in https://github.com/thiagortk/stereoVision.

**Optical Flow**: It's  computed in the code along with maskRCNN. We use the FarneBack algorithm available in OpenCV. We use the Disparity Map previously computed.
