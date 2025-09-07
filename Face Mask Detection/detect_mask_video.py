# import the necessary packages
try:
    # Try TensorFlow imports first
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
except ImportError:
    # Fall back to standalone Keras
    from keras.applications.mobilenet_v2 import preprocess_input
    from keras.preprocessing.image import img_to_array
    from keras.models import load_model

from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		try:
			preds = maskNet.predict(faces)
		except Exception as e:
			print(f"Prediction error: {str(e)}")
			# Try direct call instead
			try:
				preds = maskNet(faces).numpy()
			except Exception as e2:
				print(f"Direct call error: {str(e2)}")
				preds = []

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
import os
import tensorflow as tf

# Try to load from various model formats
model_found = False

# Try .h5 format first (most compatible)
if os.path.exists("mask_detector.h5"):
    print("[INFO] Loading model from mask_detector.h5")
    try:
        maskNet = tf.keras.models.load_model("mask_detector.h5")
        model_found = True
        print("[INFO] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load .h5 model: {str(e)}")

# Try .keras format next (for Keras 3)
if not model_found and os.path.exists("mask_detector.keras"):
    print("[INFO] Loading model from mask_detector.keras")
    try:
        maskNet = tf.keras.models.load_model("mask_detector.keras")
        model_found = True
        print("[INFO] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load .keras model: {str(e)}")

# If no model found or loading failed
if not model_found:
    print("[ERROR] No compatible model found!")
    print("Please run train_mask_detector.py first to generate a model.")
    print("Command: python train_mask_detector.py")
    exit(1)

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		
		# Handle different prediction formats
		try:
			if isinstance(pred, np.ndarray):
				if len(pred.shape) > 0 and pred.shape[0] >= 2:
					# Standard format [mask_prob, without_mask_prob]
					mask = pred[0]
					withoutMask = pred[1]
				else:
					# Single value format
					mask = pred[0]
					withoutMask = 1 - mask
			else:
				# Handle other formats (like tensors)
				mask = float(pred[0])
				withoutMask = 1 - mask
		except:
			# Fallback to binary classification
			if hasattr(pred, "__len__"):
				mask = 0.99 if pred[0] > 0.5 else 0.01
			else:
				mask = 0.99 if pred > 0.5 else 0.01
			withoutMask = 1 - mask

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# When saving the model:
# model.save("mask_detector.h5")  # Save as .h5 file
# OR
# model.save("mask_detector.keras")  # Save as .keras file