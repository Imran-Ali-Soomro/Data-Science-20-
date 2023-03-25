import numpy as np
import cv2
import time
import tensorflow as tf
import argparse

# Set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", required=True,
	help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Initialize Pi camera module
cam = cv2.VideoCapture("d:/Fiverr_Projects/Object_detection_tllite/14.jpg")
time.sleep(2.0)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=args["model"])
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load label map
with open(args["labels"], "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Run object detection on each camera frame
while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Resize frame to input size of model
    resized_frame = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))

    # Preprocess input frame
    input_data = np.expand_dims(resized_frame, axis=0)
    input_data = (input_data - input_details[0]['quantization'][0]) / input_details[0]['quantization'][1]

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Filter out weak detections
    detections = []
    for i in range(output_data.shape[1]):
        confidence = output_data[0, i, 2]
        if confidence > args["confidence"]:
            label_id = int(output_data[0, i, 1])
            label = labels[label_id]
            bbox = output_data[0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            bbox = bbox.astype(int)
            detections.append((label, confidence, bbox))

    # Draw bounding boxes on frame
    for detection in detections:
        label, confidence, bbox = detection
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, "{}: {:.2f}".format(label, confidence), (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF==ord("q")
    if key == ord("q"):
        break   