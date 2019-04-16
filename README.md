# maskRCNN with DisparityMap and OpicalFlow as post-processing
Using Disparity Map and Optical Flow info to extract features from the detected objects by maskRCNN.

**maskRCNN** example edited from: https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/

**Download and extract the model files**: <br/>
wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_v2_coco_2018_01_28.tar.gz <br/>
tar zxvf mask_rcnn_inception_v2_coco_2018_01_28.tar.gz

**Disparity Map**: We use here only the resulted Disparity Map image (see: https://github.com/thiagortk/stereoVision)

**Optical Flow**: It's in the code along with maskRCNN. We use the FarneBack algorithm available in OpenCV.
