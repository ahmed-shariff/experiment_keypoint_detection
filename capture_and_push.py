import time
import zmq
import cv2
import numpy as np
​
###########################
# define the communication
###########################
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
​
​
# define the function for TrackBar
# the function will do nothing
def TrackFun(x):
    # pass
    pass

​
# function to capture the points to do the perspective transformation.
def mouse_capturing(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left click")
        # print(x,y)
        # Adding the point to the list make sure to click the points in this order
        # top left, top right, buttom left, then buttom right
        pointList.append((x, y))
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        print("Double click")
        # print(x, y)
        pointList.append((x, y))
​
# define the camera
cap = cv2.VideoCapture(1)
​
# creating the windows
# 1- Create the window that will show the camera
cv2.namedWindow("Frame")
​
# 2- Create the window that will show the ColorTrackerBar
cv2.namedWindow("ColorTrackBar")
# value for blue color
cv2.createTrackbar("L-H", "ColorTrackBar", 0, 180, TrackFun)
cv2.createTrackbar("L-S", "ColorTrackBar", 68, 255, TrackFun)
cv2.createTrackbar("L-V", "ColorTrackBar", 154, 255, TrackFun)
cv2.createTrackbar("U-H", "ColorTrackBar", 180, 180, TrackFun)
cv2.createTrackbar("U-S", "ColorTrackBar", 255, 255, TrackFun)
cv2.createTrackbar("U-V", "ColorTrackBar", 243, 255, TrackFun)
​
​
# attache the mouse event to the frame
cv2.setMouseCallback("Frame", mouse_capturing)
# store the 4 points that will be used in the perspective transformation.
pointList = []
# Defult value for the first point for prespctive transormed
pts1 = np.float32([[155, 120], [480, 120], [20, 475], [620, 475]])
​
# going over each frame.
while True:
​    _, frame = cap.read()
    # Size of the video
    FramWidth = frame.shape[1]
    FramHeight = frame.shape[0]
​
    # let us loop over the point to make the prespective transfare.
    for center_position in pointList:
        cv2.circle(frame, center_position, 5, (0, 0, 255), -1)
​

    # Perspective transformation
    if len(pointList) >= 4:
        pts1 = np.float32([[pointList[0]], [pointList[1]], [pointList[2]], [pointList[3]]])

        pts2 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (FramWidth, FramHeight))
​
    # Making tracker to adjust the hsv color
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L-H", "ColorTrackBar")
    l_s = cv2.getTrackbarPos("L-S", "ColorTrackBar")
    l_v = cv2.getTrackbarPos("L-V", "ColorTrackBar")
    u_h = cv2.getTrackbarPos("U-H", "ColorTrackBar")
    u_s = cv2.getTrackbarPos("U-S", "ColorTrackBar")
    u_v = cv2.getTrackbarPos("U-V", "ColorTrackBar")
​
    # Define the range of the red color - revisit this one.
    # red_rang
    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])
​
​
    # define the font
    font = cv2.FONT_HERSHEY_COMPLEX
​
    # define the mask for the color red with lower red value and the upper red value
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # let enhance the mask by removing noise.
    # create a black square using numpy
    kernel = np.ones((5,5), np.uint8)
    # Debug: show and check the kernel remove it
    # cv2.imshow("Kernel", kernel). Now let us apply it to the mask
    mask = cv2.erode(mask, kernel)
    result2 = cv2.bitwise_and(frame,frame,mask=mask)
​
​
​
    # Contour detection form the mask
    # Using cv2.findContours and passing the mask
    _, countours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
​
    # know loop over contours
    for idx, cnt in enumerate(countours):
        #print(idx)
        # let  collect the area of the contours and select the one that has big area.
        area = cv2.contourArea(cnt)
        middelPoint = cv2.moments(cnt)
​
        # apply Do contours approxmation
        approx = cv2.approxPolyDP(cnt,0.02*cv2.arcLength(cnt, True), True)
        # let get the location of contour using ravel which is an array of the contour polygon
        # the top left point
        x = approx.ravel()[0]
        y = approx.ravel()[1]
​
        # let get the center of the contour
        #x1 = int(middelPoint["m10"] / middelPoint["m00"])
        #y1 = int(middelPoint["m01"] / middelPoint["m00"])
        x1,y1,w,h = cv2.boundingRect(approx)
​
        # check the area if it bigger than 400 pixcel, then draw the contour
        if area > 1400:
            cv2.drawContours(result, [approx], 0,(0,0,255),5)
            ## using the approx will tell us how many line the shape has
            #if len(approx) == 3:
            #    cv2.putText(result, "Triangle", (int((x1+w)/2),int((y1+h)/2)), font, 1, (0, 0, 0))
            #    cv2.circle(result, (int(x1 + (w / 2)), int(y1 + (h / 2))), 7, (255, 255, 255), -1)
            #elif len(approx) == 4:
            #    print("It is a rectangle")
            #    cv2.putText(result, "Rectangle" ,(int((x1+w)/2),int((y1+h)/2)),font,1,(0,0,0))
            #    cv2.circle(result, (int(x1 + (w / 2)), int(y1 + (h / 2))), 7, (255, 255, 255), -1)
            #elif 10 < len(approx) < 20:
            #    cv2.putText(result, "Circle", (int((x1+w)/2),int((y1+h)/2)), font, 1, (0, 0, 0))
            #    cv2.circle(result, (int(x1 + (w / 2)), int(y1 + (h / 2))), 7, (255, 255, 255), -1)
            #cv2.putText(result, "Circle", (int((x1 + w) / 2), int((y1 + h) / 2)), font, 1, (0, 0, 0))
            cv2.circle(result, (int(x1 + (w / 2)), int(y1 + (h / 2))), 7, (255, 255, 255), -1)
            print(idx, int(x1 + (w / 2)), int(y1 + (h / 2)))
​
​
            # comunication part start here
            ##  Wait for next request from client
            message = socket.recv()
            print("Received request: %s" % message)
​
            ##  Do some 'work'.
            ##  Try reducing sleep time to 0.01 to see how blazingly fast it communicates
            ##  In the real world usage, you just need to replace time.sleep() with
            ##  whatever work you want python to do, maybe a machine learning task?
            #time.sleep(2)
​
            ##  Send reply back to client
            ##  In the real world usage, after you finish your work, send your output here
            mymessage = str(int(x1 + (w / 2))) + "," + str(int(y1 + (h / 2)))
            socket.send_string(mymessage)
​
    # show the frames windows
    cv2.imshow("Frame", frame)
    cv2.imshow("Perspective Transformation", result)
    cv2.imshow("Mask", mask)
    #cv2.imshow("Only Object",result2)
​
​
​
    key = cv2.waitKey(1)
    if key == 27:
        break
    if key == ord("d"):
        pointList = []
​
cap.release()
cv2.destroyAllWindows()
