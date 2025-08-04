# filename: ws_yolo.py
import cv2
import numpy as np
import base64
import json
import torch
import uvicorn
from fastapi import FastAPI, WebSocket
import ultralytics

# Load YOLO model on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ultralytics.YOLO("runs/obb/train4/weights/best.pt").to(device)

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            # Decode base64 JPEG
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Run YOLO inference
            results = model.predict(img, imgsz=640)

            # Extract OBB bounding boxes
            obb_data = []
            for r in results:
                if r.obb:
                    for xywhr in r.obb.xywhr.cpu().numpy():
                        obb_data.append(xywhr.tolist())

            # Send result back as JSON
            await websocket.send_text(json.dumps({"obb": obb_data}))
        except Exception as e:
            print("Error:", e)
            break
