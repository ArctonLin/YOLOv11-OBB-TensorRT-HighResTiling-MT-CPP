import onnxruntime as ort
import numpy as np
import torch
import cv2
from pathlib import Path

# Import NMS post processing function from ultralytics
from ultralytics.utils.ops import non_max_suppression

INPUT_WIDTH = 1024
INPUT_HEIGHT = 1024

NC = 15
CLASSES = ["plane", "ship", "storage tank", "baseball diamond", "tennis court",
           "basketball court", "ground track field", "harbor", "bridge",
           "large vehicle", "small vehicle", "helicopter", "roundabout",
           "soccer ball field", "swimming pool"]

COLORS= [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
         (255, 0, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128),
         (192, 192, 192), (128, 128, 128), (64, 64, 64), (255, 165, 0),
         (75, 0, 130), (238, 130, 238)]


# === 1. Load ONNX Model ===
onnx_path = "yolo11n-obb.onnx"
session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # usually [1, 3, H, W]
print(input_shape)
input_shape = [1, 3, INPUT_HEIGHT, INPUT_WIDTH]
print(input_shape)

# === 2. Load Image and Preprocess ===
def preprocess(img_path, input_shape):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]

    # Resize and pad to square
    size = input_shape[2]
    r = size / max(h0, w0)
    new_size = (int(w0 * r), int(h0 * r))
    img_resized = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_LINEAR)
    img_padded = np.full((size, size, 3), 114, dtype=np.uint8)
    img_padded[:new_size[1], :new_size[0], :] = img_resized

    # HWC → CHW, BGR → RGB, normalize
    img_input = img_padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return img_input, r, (w0, h0), img

# === 3. Run Inference ===
def infer(image_path):
    img_input, r, orig_shape, raw_img = preprocess(image_path, input_shape)
    ort_out = session.run(None, {input_name: img_input})[0]

    # === 4. Apply NMS ===
    pred = torch.tensor(ort_out)  # shape [1, num_classes+5, num_boxes]
    #pred = pred.permute(0, 2, 1)  # shape [1, num_boxes, num_classes+5]
    #pred = pred.transpose(1, 2)   # shape [1, num_classes+5, num_boxes] like YOLOv8/YOLOv11

    print(pred.shape)

    dets = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.5, rotated=True, nc=NC)

    for det in dets[0].cpu().numpy():  # Assuming single image batch
        #print(det)
        x, y, w, h, conf, cls, angle = det
        draw_rotated_box(raw_img, (x, y, w, h, angle), cls, conf)

    return raw_img

def draw_label_with_adaptive_text(img, text, x, y, cls_id, font_scale=0.5, thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    pad = 2

    # Color Setting
    bg_color = COLORS[cls_id]
    brightness = 0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    # Background Rectangle Coordinates
    top_left = (int(x), int(y) - text_height - baseline - pad)
    bottom_right = (int(x) + text_width + 2 * pad, int(y))

    # Draw Background Rectangle
    cv2.rectangle(img, top_left, bottom_right, bg_color, thickness=-1)

    # Draw Text
    cv2.putText(
        img,
        text,
        (top_left[0] + pad, bottom_right[1] - baseline),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA
    )

# === 5. Draw Rotated Bounding Boxes ===
def draw_rotated_box(img, obb, cls, conf):
    x, y, w, h, angle = obb
    angle_deg = angle * 180 / np.pi  # convert to degrees if in radians
    rect = ((x, y), (w, h), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cls_id = int(round(cls))
    #cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw center
    cv2.drawContours(img, [box], 0, COLORS[cls_id], 3)
    #cv2.putText(img, f'{CLASSES[cls_id]}, {conf:.2f}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls_id], 2)
    draw_label_with_adaptive_text(img, f'{CLASSES[cls_id]}, {conf:.2f}', x, y, cls_id)

# === 6. Run and Show ===
if __name__ == "__main__":
    result = infer("boats1024.jpg")
    cv2.imshow("YOLOv10-OBB Result", result)
    cv2.waitKey(0)
