# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:27:12 2020

@author: Aditya Saxena
"""

from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
 
 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="Minimum area size")
args = vars(ap.parse_args())
 
# If the video argument is None, let us start reading from the web cam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
 
# Otherwise, if we have the video file argument, let us read from the video file
else:
	vs = cv2.VideoCapture(args["video"])
 
# Initialise the first frame in the video
firstFrame = None
 
# Start looping over the frames of the video/webcam
while True:
	# Grab the current frame and initialise the occupied/unoccupied text
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "No motion"
 
	# If we were not able to read the file, then we have reached the end of the video
	if frame is None:
		break
 
	# Resize the frame, convert it to grayscale and blur it
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# If the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
 
	# Compute the absolute difference between the current frame and the first frame that we have already stored
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
	# Dilate the threshold image to fill in holes, then find contours on the thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
 
	# Now that we have all the contours, let us loop over the contours
	for c in cnts:
		# If the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
 
		# Compute the bounding box of the contour and draw it on the frame. Update the text accordingly
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		text = "Motion is detected"
 
	# Draw the text and the timestamp of when it happened on the frame
	cv2.putText(frame, "Motion status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0]-10), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# Show the frame and record if the user presses a key
	cv2.imshow("Video feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame delta", frameDelta)
 
	key = cv2.waitKey(1) & 0xFF
	# If the 'q' key is pressed, break from the loop
	if key == ord('q'):
		break
 
# Clean up the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()