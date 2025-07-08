
trtexec --onnx=yolo11n-obb.onnx --saveEngine=yolo11n-obb-fp16.trt --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --minShapes=images:1x3x160x160 --optShapes=images:16x3x1024x1024 --maxShapes=images:24x3x1280x1280

pause
