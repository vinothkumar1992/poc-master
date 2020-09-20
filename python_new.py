# import the necessary packages
import numpy as np
import argparse
import cv2
from shapely.geometry import Polygon
import time
import threading
import math

def task(video,prototxt,model):
  print("Timer object is getting executed...*****************************************")

  t = threading.Timer(1, task)
  temp=0

	# construct the argument parse and parse the arguments
#   ap = argparse.ArgumentParser()

#   ap.add_argument("-c", "--confidence", type=float, default=0.2,
# 		help="minimum probability to filter weak detections")
  conf=0.2
#   args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
  CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
  COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

	# load our serialized model from disk
  net = cv2.dnn.readNetFromCaffe(prototxt, model)
  print(prototxt)
  print(model)


	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	# (note: normalization is done via the authors of the MobileNet SSD
	# implementation)
	# image = cv2.imread(args["image"])
  cap = cv2.VideoCapture(video)
  counter=0
  timer=0
  frequency_error=0
  ret, image = cap.read()
  skip_frames=1
  illegal=False
  alert=False
  illegal_detection_percent=20
  alarm_time=2
  print("Testing")

  while (ret):
	  counter+=1
	  ret, image = cap.read()
	  if counter%skip_frames==0:
			# print(counter)
			# ret, image = cap.read()
			# 
		  
		  (h, w) = image.shape[:2]
			# cv2.imwrite("C:\Users\qmobi\Desktop\Python\Object-Detection-Tutorial-master\images\parking.jpg", image)
		  blob = cv2.dnn.blobFromImage(cv2.resize(image, (318, 260)), 0.007843, (300, 300), 127.5)

			#draw polygon
			# top-left, buttom-left, buttom-right, top-right
		  
		  pts = np.array([[250,130],[230,180],[380,205],[405,150]], np.int32)
          
		  pts_reshape = pts.reshape((-1,1,2))
		  image = cv2.polylines(image,[pts_reshape],True,(0,255,255))
		  
		#   print("[INFO] loading 4751546...")	

			# pass the blob through the net work and obtain the detections and predictions
			# print("[INFO] computing object detections...")
		  net.setInput(blob)
		  detections = net.forward()
			# loop over the detections
			# print( detections.shape[2])
			# print(detections.shape[2])
		  illegal=False
		  for i in np.arange(0, detections.shape[2]):
				
				
				# extract the confidence (i.e., probability) associated with the prediction
			   confidence = detections[0, 0, i, 2]
				# filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
			   if confidence > conf:
					# extract the index of the class label from the `detections`,
					# then compute the (x, y)-coordinates of the bounding box for the object
				   idx = int(detections[0, 0, i, 1])
					
				   if(idx==7 ): #dedtect only cars 

					   box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
					   (startX, startY, endX, endY) = box.astype("int")

						#calculate polygon coordinatef from detection points
					   ax=startX
					   ay=startY
					   bx=startX
					   by=endY
					   cx=endX
					   cy=endY
					   dx=endX
					   dy=startY
						
					   detected_pts = np.array([[ax,ay],[bx,by],[cx,cy],[dx,dy]], np.int32)
						# detected_pts_reshape = detected_pts.reshape((-1,1,2))
						# image = cv2.polylines(image,[detected_pts_reshape],True,(0,0,255))

						# find collision ````````````````````````````````````````````

					   p1=Polygon(pts)
					   p2=Polygon(detected_pts)
					   p3=p2.intersection(p1)
						# print(p3) # result: POLYGON ((0.5 0.5, 1 1, 1 0, 0.5 0.5))
					   detection_percentage=p3.area*100/p1.area
						# print(detection_percentage,"%") # result: 0.25

					#    illegal=False
					   if(detection_percentage>=illegal_detection_percent):
						   illegal=True
					   
							
							# temp+=skip_frames
							# if temp>150:
							# 	print("*********************Car is on wrong parking**************************")
						# else:
						# 	temp=0


						# ``````````````````````````````````````````````````````````

						# display the prediction
					   label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
						# print("[INFO] {}".format(label))
					   cv2.rectangle(image, (startX, startY), (endX, endY),
							COLORS[idx], 2)
					   y = startY - 15 if startY - 15 > 15 else startY + 15
					   cv2.putText(image, label+" Illegal:"+str(detection_percentage)[0:2]+"%", (startX, y),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
						

				# end of if condition
				# Define Alarm Time in seconds
				# alarm_time=5
		  time_check=(30*alarm_time)/skip_frames

		#   print("time",timer)
		
		  if illegal==True:
			   frequency_error=0
			   timer+=1
			# illegal=False
		#    print("ttttttttttttttttttttttttttttttttttttttttttttttttttttttttt")	
			   if math.floor(timer)>=time_check:

				   alert=True
				#    print("************ wrong Parking ********",math.floor(timer/100),time_check)
				   illegal=True
		  else:
			   frequency_error+=1
			#    timer=0
			   timer+=1
			#    print("+++++  ELSE TIMER ",timer,"frequency timer ",frequency_error)
			   if frequency_error>=5:
				   illegal=False
				   alert=False
	
				   timer=0
					  
	
						


			# show the output image
		  if alert==True:

			  cv2.putText(image, "Illegal Parking Alert !!!******", (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
			#   print("Illegal Parking Alert !!!******")
			#   cv2.imshow("Output", image)
			  frames = cv2.imencode('.jpg', image)[1].tobytes()
			  yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')
		
			  cv2.waitKey(1)	
		  else:
		
			  cv2.putText(image, "OK", (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
			  frames = cv2.imencode('.jpg', image)[1].tobytes()
			  yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frames + b'\r\n')
			  cv2.waitKey(1)




	

# USAGE
# python deep.py --video images/example_01.jpg --prototxt MobileNetSSD.prototxt.txt --model NetSSD.caffemodel
# if __name__ == "__main__":
# 	task("images/parking.mp4","MobileNetSSD.prototxt.txt","NetSSD.caffemodel")

