# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import pytesseract
import requests
import imutils

url = "http://text-processing.com/api/sentiment/"


from nets import net

def request_sentiment(text):
	params = {"text":text}
	r = requests.post(url, data=params)
	return r.json()

def get_ocr(image):
	text = pytesseract.image_to_string(image, lang="eng")
	return text

def invert(image):
	return (255-image)

def get_meme_text(image):
	text = ''
	# load the input image and grab the image dimensions
	# image = cv2.imread(image_path)
	hh, ww, _ = image.shape
	diff = ww - hh
	# print(hh, ww)
	if diff > 0:
		black = np.zeros((diff, ww, 3), np.uint8)
		# print(black.shape)
		image = np.append(image, black, axis=0)
	elif diff < 0:
		black = np.zeros((hh, ww-diff, 3), np.uint8)
		black[:, :ww, :] = image
		image = black

	# cv2.imshow('', image)
	orig = image.copy()
	super_orig = image.copy()

	(H, W) = image.shape[:2]

	# set the new width and height and then determine the ratio in change
	# for both the width and height

	(newW, newH) = (320, 320)
	# (newW, newH) = (640, 640)
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]



	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])


	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	if len(boxes) > 0:
		boxes = boxes[boxes[:,1].argsort()]
	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)
		# off = 10

		# startX -= off
		# startY -= off
		# endX += off
		# endY += off
		w = (endX-startX)
		h = (endY-startY)
		# white_canvas = np.ndarray([255, 255, 255]*w)

		white_canvas = np.zeros((h*3, w*3), np.uint8)
		white_canvas[:] = [255]
		crop_img = super_orig[startY:endY, startX:endX]

		gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
		# blur = cv2.GaussianBlur(gray, (5, 5), 0)
		ret2, th2 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


		inv_bin = invert(th2)
		# inv_bin = invert(crop_img)
		white_canvas[h:h*2, w:w*2, ] = inv_bin
		text += ' ' + get_ocr(white_canvas)

	return text


def meme_info(image):
	text = get_meme_text(image)
	label = ''
	if text != '':
		label = request_sentiment(text)['label']
	# print(sentiments['label'])
	return text, label

