"""
Import models
"""

from keras.models import load_model
import cv2
east_model = "frozen_east_text_detection.pb"
net = cv2.dnn.readNet(east_model)
model = load_model("weights.hdf5")
print("Models ready")
