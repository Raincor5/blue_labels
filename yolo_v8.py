import ultralytics

# Load model
model = ultralytics.YOLO(r"runs\obb\train4\weights\best.pt")

# Predict
results = model.predict(
    source=r"P:\CS-ML-DS-etc\OCR-LABELS\dataset\test\images\IMG_1344_JPEG.rf.67713b71a64f02ee173290287219ac39.jpg",
    save=True
)

# Access OBB results
for result in results:
    obb = result.obb
    print("OBB xywhr:", obb.xywhr)  # center_x, center_y, width, height, rotation
    print("OBB xyxy:", obb.xyxy)    # four corner points of each rotated box
