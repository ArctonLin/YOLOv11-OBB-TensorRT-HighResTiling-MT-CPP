from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model

# Export the model
model.export(format="onnx", dynamic=True)