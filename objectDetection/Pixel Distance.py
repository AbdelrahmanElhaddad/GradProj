import cv2
import math
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # Pre-trained model

# Perform detection
img = cv2.imread('test.jpg')
results = model('test.jpg')  # Replace with your image/video path

# Extract bounding boxes
detections = results[0].boxes  # First result (image)

cars = []
humans = []

# Filter detections for cars and humans
for box in detections:
    class_id = int(box.cls[0])
    if class_id == 2:  # Car
        cars.append((box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))
    elif class_id == 0:  # Human
        humans.append((box.xyxy[0][0], box.xyxy[0][1], box.xyxy[0][2], box.xyxy[0][3]))

# Calculate distances
distances = []
for car in cars:
    car_center = ((car[0] + car[2]) / 2, (car[1] + car[3]) / 2)
    for human in humans:
        human_center = ((human[0] + human[2]) / 2, (human[1] + human[3]) / 2)
        # Calculate Euclidean distance
        distance = math.sqrt((human_center[0] - car_center[0]) ** 2 + 
                             (human_center[1] - car_center[1]) ** 2)
        
        distances.append(distance)

# Output distances
for i, distance in enumerate(distances):
    print(f"Distance {i+1}: {distance} pixels")


for car in cars:
    car_center = ((car[0] + car[2]) / 2, (car[1] + car[3]) / 2)
    cv2.circle(img, (int(car_center[0]), int(car_center[1])), 5, (255, 0, 0), -1)  # Car: Blue

for human in humans:
    human_center = ((human[0] + human[2]) / 2, (human[1] + human[3]) / 2)
    cv2.circle(img, (int(human_center[0]), int(human_center[1])), 5, (0, 255, 0), -1)  # Human: Green


for car in cars:
    car_center = (int((car[0] + car[2]) / 2), int((car[1] + car[3]) / 2))
    for human in humans:
        human_center = (int((human[0] + human[2]) / 2), int((human[1] + human[3]) / 2))
        cv2.line(img, human_center, car_center, color=(255, 255, 255), thickness=2)
        
cv2.imshow("Detections", img)
cv2.waitKey(0)

