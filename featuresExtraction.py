import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
maskThreshold = 0.3  # Mask threshold

parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN object detection and segmentation')
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask, imghsv, flow, gray):
    color = colors[classId%len(colors)]
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), color, 1)
    
    # Print a label of class.
    label = (classes[classId])
    
    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 0.65, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(labelSize[1])), (left + round(labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,0), 1)

    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = imghsv[top:bottom+1, left:right+1][mask]

    # Draw mask with Dispatry values
    frame[top:bottom+1, left:right+1][mask] = cv.addWeighted(frame[top:bottom+1, left:right+1][mask],0.3,imghsv[top:bottom+1, left:right+1][mask],0.7,0)

    # --------------------------------------------- #
    # ------ Disparity Map Average Intensity ------ #
    # --------------------------------------------- #
    grayRoi = cv.cvtColor(imghsv, cv.COLOR_BGR2GRAY)

    # Contours on the image
    mask = mask.astype(np.uint8)
    im2, contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # Calculate the distances to the contour
    raw_dist = np.empty(mask.shape, dtype=np.float32)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            raw_dist[i,j] = cv.pointPolygonTest(contours[0], (j,i), True)

    # Depicting the distances graphically
    veCount = 0
    intensityCount = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if raw_dist[i,j] >= 0:
                intensity = grayRoi[top+i,left+j]
                veCount = veCount + 1
                intensityCount = intensityCount + intensity
    intensityM = int(round(intensityCount/veCount))

    # Labels intensity classes
    if intensityM >= 185:
        depthLabel = 'very close'
    elif intensityM >= 115 and intensityM < 185:
        depthLabel = 'close'
    elif intensityM >= 45 and intensityM < 115:
        depthLabel = 'far'
    elif intensityM < 45:
        depthLabel = 'very far'

    '''# Display the depthLabel at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(depthLabel, cv.FONT_HERSHEY_PLAIN, 0.65, 1)
    top = max(top+10, labelSize[1])
    cv.rectangle(frame, (left, top - round(labelSize[1])), (left + round(labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, depthLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,0), 1)
    #cv.putText(frame, depthLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,255), 1)'''

    # ---------------------------------------- #
    # ---------- END INTENSITY --------------- #
    # ---------------------------------------- #

    # ------ Optical Flow ------ #

    step = 16

    # --------------------------------------------- #
    # ------ Optical Flow in the whole frame ------ #
    # --------------------------------------------- #
    '''h, w = frame.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    cv.polylines(frame, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(frame, (_x2, _y2), 1, (0, 255, 0), -1)'''
    # ---------------------------------------------------- #
    # ------ End of Optical Flow in the whole frame ------ #
    # ---------------------------------------------------- #
    
    # ---------------------------------- #
    # ------ OF only in he object ------ #
    # ---------------------------------- #
    '''for y in range(top,bottom,step):
        for x in range(left,right,step):
            fx, fy = flow[y,x].T
            cv.line(frame, (x,y), (int(round(x+fx)), int(round(y+fy))), (0,255,0))
            cv.circle(frame, (int(round(x+fx)), int(round(y+fy))), 1, (0,255,0), -1)'''
    # ----------------------------------- #
    # ------ OF ends in the object ------ #
    # ----------------------------------- #

    # ------------------------------------------------------ #
    # ------ OF in the object average and centralized ------ #
    # ------------------------------------------------------ #
    ofCount = 0
    fxCount = 0
    fyCount = 0
    for y in range(top,bottom,step):
        for x in range(left,right,step):
            fx, fy = flow[y,x].T
            ofCount = ofCount + 1
            fxCount = fxCount + fx
            fyCount = fyCount + fy
    
    fxM = fxCount/ofCount
    fyM = fyCount/ofCount
    
    flowVectorLength = fxM * fyM
    #print(" --> Flow Length --> ", abs(flowVectorLength))

    objCenterW = int(round((left+right)/2))
    objCenterH = int(round((top+bottom)/2))

    cv.line(frame, (objCenterW,objCenterH), (int(round(objCenterW+fxM)), int(round(objCenterH+fyM))), (255,0,0), 1)
    cv.circle(frame, (int(round(objCenterW+fxM)), int(round(objCenterH+fyM))), 1, (255,0,0), -1)
    # ---------------------------------------------------------------- #
    # ------ OF ends in the object with average and centralized ------ #
    # ---------------------------------------------------------------- #

    # Vectors labels information classes
    if fxM >= 1:
        flowDirHLabel = 'left to right'
    elif fxM <= -1:
        flowDirHLabel = 'right to left'
    elif fxM < 1 and fxM > -1:
        flowDirHLabel = 'stable direction'

    if fyM >= 1:
        flowDirZLabel = 'approaching'
    elif fyM <= -1:
        flowDirZLabel = 'moving away'
    elif fyM < 1 and fyM > -1:
        flowDirZLabel = 'stable distance'

    if abs(flowVectorLength) >= 100:
        flowLengthLabel = 'very fast'
    elif abs(flowVectorLength) >= 5 and flowVectorLength < 100:
        flowLengthLabel = 'fast'
    elif abs(flowVectorLength) >= 0.5 and abs(flowVectorLength) < 5:
        flowLengthLabel = 'average speed' 
    elif abs(flowVectorLength) >= 0.1 and abs(flowVectorLength) < 0.5:
        flowLengthLabel = 'slow'
    elif abs(flowVectorLength) < 0.1:
        flowLengthLabel = 'stopped'

    '''# Draw flow infos
    # Display the flows infos at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(flowDirHLabel, cv.FONT_HERSHEY_PLAIN, 0.65, 1)
    top = max(top+10, labelSize[1])
    cv.rectangle(frame, (left, top - round(labelSize[1])), (left + round(labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, flowDirHLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,0), 1)
    #cv.putText(frame, flowDirHLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,255), 1)

    labelSize, baseLine = cv.getTextSize(flowDirZLabel, cv.FONT_HERSHEY_PLAIN, 0.65, 1)
    top = max(top+10, labelSize[1])
    cv.rectangle(frame, (left, top - round(labelSize[1])), (left + round(labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, flowDirZLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,0), 1)
    #cv.putText(frame, flowDirZLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,255), 1)

    labelSize, baseLine = cv.getTextSize(flowLengthLabel, cv.FONT_HERSHEY_PLAIN, 0.65, 1)
    top = max(top+10, labelSize[1])
    cv.rectangle(frame, (left, top - round(labelSize[1])), (left + round(labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, flowLengthLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,0), 1)
    #cv.putText(frame, flowLengthLabel, (left, top), cv.FONT_HERSHEY_PLAIN, 0.65, (0,0,255), 1)'''

    # ------ OF ENDS ------ #

# For each frame, extract the bounding box and mask for each detected object
def postprocess(boxes, masks, imghsv, flow, gray):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    frameH = imghsv.shape[0]
    frameW = imghsv.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
            
            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])
            
            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))
            
            # Extract the mask for the object
            classMask = mask[classId]

            # Draw bounding box, colorize and show the mask on the image
            drawBox(frame, classId, score, left, top, right, bottom, classMask, imghsv, flow, gray)


# Load names of classes
classesFile = "mscoco_labels.names";
classes = None
with open(classesFile, 'rt') as f:
   classes = f.read().rstrip('\n').split('\n')

# Give the textGraph and weight files for the model
textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

# Load the network
net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph);
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# Load the classes
colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = [] #[0,0,0]
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

winName = 'Objects features extraction'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "mask_rcnn_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_mask_rcnn_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_mask_rcnn_out_py.avi'
else:
    # Webcam input
    #cap = cv.VideoCapture(0)
    #cap = cv.VideoCapture("KITTI_Germany/City/City/2011_09_26_6/image_02/data/%10d.png")
    cap = cv.VideoCapture("CARINA_Brasil/Easy/easy_stereo_narrow_left/easy4_stereo_narrow_left/%09d.png")
    outputFile = 'result.avi'
    capDisparity = cv.VideoCapture("MDHOT.avi") #Get the disparitymap frames (or video)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 15, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

width = cap.get(3) #P/ OF
height = cap.get(4) #P/ OF

ret, prev = cap.read()
prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

while cv.waitKey(1) < 0:
    
    # Get frame from the video
    hasFrame, frame = cap.read()
    hasImghsv, imghsv = capDisparity.read()    

    # Stop the program if reached end of video
    if not hasFrame or not hasImghsv:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    #P/ OF
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    #END P/ OF

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run the forward pass to get output from the output layers
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # Extract the bounding box and mask for each of the detected objects
    postprocess(boxes, masks, imghsv, flow, gray)

    # Put efficiency information.
    t, _ = net.getPerfProfile()
    '''label = 'Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))'''

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)
