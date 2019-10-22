# USAGE
# python object_tracker.py -v 20190923_144136.mp4
# python object_tracker.py -v 20191010_155216.mp4 -f 30
# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import argparse
import time
import cv2
import os
import keyboard
from datetime import datetime

from test import TestKeypointRcnn, torch_tensor_to_img


# function to capture the point from the frame.
class MouseCapture():
    def __init__(self, pointList):
        self.pointList = pointList

    def __call__(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print("Left click")
            self.pointList.append((x, y))

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # print("Double click")
            self.pointList.append((x, y))

def main():
    # start without log the data
    loggingPrespective = False
    loggingNormal = False
    mouse_capturing = MouseCapture([])
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video_source", required=False, default='/dev/video0',
                    help="The video to use as a source")
    ap.add_argument("-f", "--framesToEscape", required=False, default=30,
                    help="Capture data every 30 frames")
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-m", "--model", required=False,
                    help="path to Caffe pre-trained model")
    args = vars(ap.parse_args())

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)

    # create file to write the data to
    if os.path.exists("datasetPoints.csv"):
        dataPoints = open("datasetPoints.csv", "w")
    else:
        dataPoints = open("datasetPoints.csv", "w+")

    dataPoints.write('Datetime,ObjectID,xLocation,yLocation\n')

    # load our serialized model from disk
    print("[INFO] loading model...")
    out_size = 1000
    # Defult value for the first point for prespctive transormed
    pts1 = np.float32([[155, 120], [480, 120], [20, 475], [620, 475]])
    pts1 = np.float32([[0, 0], [out_size, 0], [0, out_size], [out_size, out_size]])

    # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    net = TestKeypointRcnn(920, out_size)

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args['video_source'])
    framecount = 0

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.

    if int(major_ver) < 3:
        fps = vs.get(cv2.cv.CV_CAP_PROP_FPS)
        # FrameRate = cv2.VideoCapture(args['framesToEscape'])
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = vs.get(cv2.CAP_PROP_FPS)
        # FrameRate = cv2.VideoCapture(args['framesToEscape'])
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # create the frame windows
    cv2.namedWindow("Frame")
    # attach the mouse call bace event to the windows
    cv2.setMouseCallback("Frame", mouse_capturing)
    # store the 4 points that will be prespctive transormed in the list
    #time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # read the next frame from the video stream and resize it
        # time.sleep(2.0)
        r, frame = vs.read()
        framecount += 1
        result = frame
        # frame = imutils.resize(frame, width=800)

        for center_position in mouse_capturing.pointList:
            # print(center_position)
            cv2.circle(frame, center_position, 5, (0, 0, 255), -1)

        # # construct a blob from the frame, pass it through the network,
        # # obtain our output predictions, and initialize the list of
        # # bounding box rectangles
        # blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
        #                              (104.0, 177.0, 123.0))

        predictions, frame = net(frame)
        rects = []
        frame = torch_tensor_to_img(frame)
        result = frame.copy()

        # if the frame dimensions are None, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            H = int(H/out_size)
            W = int(W/out_size)

        ################################################################
        # here where the prespective transformation should happen
        # the following link is showing that we can apply to a point
        #
        ################################################################
        if len(mouse_capturing.pointList) >= 4:
            pts1 = np.float32([[mouse_capturing.pointList[0]],
                               [mouse_capturing.pointList[1]],
                               [mouse_capturing.pointList[2]],
                               [mouse_capturing.pointList[3]]])
            # print("point0:", [pointList[0]], "point1:", [pointList[1]], "point2:", [pointList[2]], "point3:",[pointList[3]])

        pts2 = np.float32([[0, 0],
                           [result.shape[1], 0],
                           [0, result.shape[0]],
                           [result.shape[1], result.shape[0]]])
        # calculate the the prepective transformation materix.
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # apply the transformation to the result frame
        result = cv2.warpPerspective(frame, matrix, result.shape[:2])

        # loop over the detections
        for i in range(0, predictions['boxes'].shape[0]):
            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if predictions['scores'][i] > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = predictions['boxes'][i].cpu().detach().numpy()
                rects.append(box.astype("int"))

                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = (box * np.array([W, H, W, H])).astype("int")
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

            # captured data after 30 frame
            if framecount >= (fps * 10):
                print('Frame number', framecount)
                framecount = 0

                if (loggingNormal):
                    dataPoints.write(
                    '{0},{1},{2},{3}\n'.format(datetime.now(), objectID, centroid[0], centroid[1]))

                # Print the object time, object ID , x, y
                # These points are from the orginal frame
                transformed_points = cv2.perspectiveTransform(
                    centroid.reshape(1, 1, -1).astype(np.float), matrix)[0, 0]
                # if logging is activited
                if(loggingPrespective):
                    # log the interested area
                    if transformed_points[0] > 0 and transformed_points[1] > 0 and \
                       transformed_points[0] < result.shape[0] and transformed_points[1] < result.shape[1]:
                        print(datetime.now(), ";", objectID, ";",transformed_points[0], ";", transformed_points[1])
                        # write to the file
                        dataPoints.write('{0},{1},{2},{3}\n'.format(datetime.now(),objectID,transformed_points[0],transformed_points[1]))
                        # cv2.circle(result, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                        cv2.circle(result,
                                   (int(transformed_points[0]), int(transformed_points[1])),
                                   3, (255, 100, 0), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.imshow("Result", result)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop

        if key == ord("q"):
            break
        if key == ord("d"):
            mouse_capturing.pointList = []
        if key == ord("r"):
            loggingPrespective = True
        if key == ord("n"):
            loggingNormal = True
        if key == ord("s"):
            logging = False

    # do a bit of cleanup
    cv2.destroyAllWindows()
    # close the file
    dataPoints.close()
    # vs.stop()


if __name__ == "__main__":
    main()
