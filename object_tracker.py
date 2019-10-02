# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from datetime import datetime


# function to capture the point from the frame.
def mouse_capturing(event, x,y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("Left click")
        pointList.append((x,y))

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        #print("Double click")
        pointList.append((x,y))


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# create the frame windows
cv2.namedWindow("Frame")
# attach the mouse call bace event to the windows
cv2.setMouseCallback("Frame", mouse_capturing)
# store the 4 points that will be prespctive transormed in the list
pointList = []
# Defult value for the first point for prespctive transormed
pts1 = np.float32([[155,120], [480, 120], [20, 475], [620, 475]])


# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# read the next frame from the video stream and resize it
	frame = vs.read()
	result = frame
	frame = imutils.resize(frame, width=400)

	for center_position in pointList:
		# print(center_position)
		cv2.circle(frame, center_position, 5, (0, 0, 255), -1)

	# if the frame dimensions are None, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the frame, pass it through the network,
	# obtain our output predictions, and initialize the list of
	# bounding box rectangles
	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
		(104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()
	rects = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# filter out weak detections by ensuring the predicted
		# probability is greater than a minimum threshold
		if detections[0, 0, i, 2] > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object, then update the bounding box rectangles list
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
			rects.append(box.astype("int"))

			# draw a bounding box surrounding the object so we can
			# visualize it
			(startX, startY, endX, endY) = box.astype("int")
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 2)

	# update our centroid tracker using the computed set of bounding
	# box rectangles
	objects = ct.update(rects)

	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		################################################################
		# here where the prespective transformation should happen
		# the following link is showing that we can apply to a point
		#
		################################################################
		if len(pointList) >= 4:
			pts1 = np.float32([[pointList[0]], [pointList[1]], [pointList[2]], [pointList[3]]])
			#print("point0:", [pointList[0]], "point1:", [pointList[1]], "point2:", [pointList[2]], "point3:",[pointList[3]])

		pts2 = np.float32([[0, 0], [frame.shape[1], 0], [0, frame.shape[0]], [frame.shape[1], frame.shape[0]]])
		# calculate the the prepective transformation materix.
		matrix = cv2.getPerspectiveTransform(pts1, pts2)
		# apply the transformation to the result frame
		result = cv2.warpPerspective(frame, matrix, (W, H))

		cv2.circle(result, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		# Print the object time, object ID , x, y
		# These points are from the orginal frame
		print(datetime.now(),";",objectID,";",centroid[0],";",centroid[1])

		# THE QUESTION HOW TO TRANSFORM GET THE PREPECTIVE TRANSFORMATION FOR THE ABOVE POINTS?
		# HERE WHERE I NEED HELP WITH


	# show the output frame
	cv2.imshow("Frame", frame)
	cv2.imshow("Result", result)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()