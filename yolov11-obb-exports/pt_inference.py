from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model

# Predict with the model
results = model("boats1024.jpg")  # predict on an image


# Access the results
for result in results:
    xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
    xyxyxyxy = result.obb.xyxyxyxy  # polygon format with 4-points
    names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
    confs = result.obb.conf  # confidence score of each box
    print(xywhr)

img = results[0].plot()  # returns a numpy array with drawn boxes
cv2.imshow("YOLO OBB Results", img)
cv2.waitKey(0)
cv2.destroyAllWindows()