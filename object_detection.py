import cv2
import numpy as np

# Load YOLO model and classes
weights_path = 'yolov4.weights'  # Path to YOLO weights file
config_path = 'yolov4.cfg'       # Path to YOLO configuration file

# Load YOLO network
net = cv2.dnn.readNet(weights_path, config_path)

# Load class labels (can be COCO dataset class names)
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define the image to analyze
image_path = 'input.jpg'  # Replace with your image path
image = cv2.imread(image_path)
h, w = image.shape[:2]

# Preprocess image for YOLO
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get YOLO output layer names
layer_names = net.getUnconnectedOutLayersNames()

# Forward pass through YOLO
layer_outputs = net.forward(layer_names)

# Initialize lists for detection data
boxes = []  # Bounding boxes
confidences = []  # Confidence scores
class_ids = []  # Class IDs

# Process YOLO output
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]  # Class probabilities
        class_id = np.argmax(scores)  # Get the class ID with the highest score
        confidence = scores[class_id]  # Get the highest score
        
        if confidence > 0.5:  # Filter detections by confidence threshold
            # Get object position and size
            center_x = int(detection[0] * w)
            center_y = int(detection[1] * h)
            width = int(detection[2] * w)
            height = int(detection[3] * h)
            
            # Calculate top-left corner of the bounding box
            x = int(center_x - width / 2)
            y = int(center_y - height / 2)
            
            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression to remove overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Count objects
object_count = len(indices)
print(f'Total objects detected: {object_count}')

# Draw bounding boxes and labels on the image
for i in indices.flatten():  # Flatten the indices to iterate through them
    x, y, width, height = boxes[i]  # Get box coordinates
    label = f"{classes[class_ids[i]]}: {int(confidences[i] * 100)}%"
    color = (0, 255, 0)  # Green color for bounding box
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)  # Draw rectangle
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Label

# Show the image with detections
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result to disk (optional)
cv2.imwrite('output.jpg', image)  # Save the output image with bounding boxes
