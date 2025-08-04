# filename: improved_ws_yolo.py
import cv2
import numpy as np
import base64
import json
import torch
import uvicorn
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import ultralytics
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOWebSocketServer:
    def __init__(self, model_path: str = "runs/obb/train4/weights/best.pt"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.active_connections: List[WebSocket] = []
        
        # Performance tracking
        self.inference_times = []
        self.total_requests = 0
        
        self.load_model()
        
    def load_model(self):
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
            self.model = ultralytics.YOLO(self.model_path).to(self.device)
            logger.info("Model loaded successfully")
            
            # Get model info
            if hasattr(self.model, 'names'):
                logger.info(f"Model classes: {self.model.names}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message with server info
        welcome_msg = {
            "type": "info",
            "message": "Connected to YOLO WebSocket Server",
            "device": self.device,
            "model_path": self.model_path,
            "total_requests_processed": self.total_requests
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
        
    async def process_image(self, websocket: WebSocket, image_data: str):
        """Process image and return detection results"""
        try:
            start_time = time.time()
            
            # Decode base64 image
            img_data = base64.b64decode(image_data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to decode image"
                }))
                return
                
            decode_time = time.time()
            
            # Run YOLO inference
            results = self.model.predict(img, imgsz=640, verbose=False)
            
            inference_time = time.time()
            
            # Extract detection data
            detection_data = self.extract_detections(results)
            
            processing_time = time.time()
            
            # Calculate timing
            total_time = processing_time - start_time
            decode_duration = decode_time - start_time
            inference_duration = inference_time - decode_time
            extract_duration = processing_time - inference_time
            
            # Track performance
            self.inference_times.append(inference_duration)
            if len(self.inference_times) > 100:  # Keep last 100 measurements
                self.inference_times.pop(0)
            self.total_requests += 1
            
            # Prepare response
            response = {
                "type": "detection_result",
                "obb": detection_data["obb"],
                "regular_boxes": detection_data.get("boxes", []),
                "timing": {
                    "total_ms": round(total_time * 1000, 2),
                    "decode_ms": round(decode_duration * 1000, 2),
                    "inference_ms": round(inference_duration * 1000, 2),
                    "extract_ms": round(extract_duration * 1000, 2)
                },
                "image_info": {
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "channels": img.shape[2]
                },
                "performance": {
                    "avg_inference_ms": round(np.mean(self.inference_times) * 1000, 2),
                    "total_requests": self.total_requests
                }
            }
            
            # Send results
            await websocket.send_text(json.dumps(response))
            
            # Log performance every 50 requests
            if self.total_requests % 50 == 0:
                avg_inference = np.mean(self.inference_times) * 1000
                logger.info(f"Performance update - Total requests: {self.total_requests}, "
                          f"Avg inference time: {avg_inference:.2f}ms")
                          
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            }))
            
    def extract_detections(self, results) -> Dict[str, Any]:
        """Extract detection data from YOLO results"""
        detection_data = {
            "obb": [],
            "boxes": []
        }
        
        for r in results:
            # Extract OBB (Oriented Bounding Boxes) if available
            if hasattr(r, 'obb') and r.obb is not None:
                if len(r.obb.xywhr) > 0:
                    # Get OBB data: center_x, center_y, width, height, rotation
                    xywhr = r.obb.xywhr.cpu().numpy()
                    confs = r.obb.conf.cpu().numpy()
                    
                    # Get class info if available
                    if hasattr(r.obb, 'cls') and r.obb.cls is not None:
                        classes = r.obb.cls.cpu().numpy()
                    else:
                        classes = np.zeros(len(xywhr))
                    
                    for i, (box, conf, cls) in enumerate(zip(xywhr, confs, classes)):
                        detection_data["obb"].append({
                            "center_x": float(box[0]),
                            "center_y": float(box[1]),
                            "width": float(box[2]),
                            "height": float(box[3]),
                            "rotation": float(box[4]),
                            "confidence": float(conf),
                            "class_id": int(cls),
                            "class_name": self.model.names.get(int(cls), f"Class_{int(cls)}")
                        })
            
            # Extract regular bounding boxes if available
            if hasattr(r, 'boxes') and r.boxes is not None:
                if len(r.boxes.xyxy) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confs, classes):
                        detection_data["boxes"].append({
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                            "confidence": float(conf),
                            "class_id": int(cls),
                            "class_name": self.model.names.get(int(cls), f"Class_{int(cls)}")
                        })
        
        return detection_data

# Create FastAPI app
app = FastAPI(title="YOLO WebSocket Server", version="1.0.0")

# Initialize YOLO server
yolo_server = YOLOWebSocketServer()

@app.get("/")
async def get():
    """Serve a simple test page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO WebSocket Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
            pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YOLO WebSocket Server</h1>
            
            <div class="status success">
                ✅ Server is running and ready to accept connections
            </div>
            
            <div class="info">
                <h3>Server Information:</h3>
                <ul>
                    <li><strong>WebSocket URL:</strong> ws://localhost:8000/ws</li>
                    <li><strong>Device:</strong> """ + yolo_server.device + """</li>
                    <li><strong>Model:</strong> """ + yolo_server.model_path + """</li>
                    <li><strong>Active Connections:</strong> <span id="connections">""" + str(len(yolo_server.active_connections)) + """</span></li>
                    <li><strong>Total Requests:</strong> """ + str(yolo_server.total_requests) + """</li>
                </ul>
            </div>
            
            <h3>Usage:</h3>
            <p>Connect to the WebSocket endpoint and send base64-encoded JPEG images for detection.</p>
            
            <h4>Example Response Format:</h4>
            <pre>{
  "type": "detection_result",
  "obb": [
    {
      "center_x": 320.5,
      "center_y": 240.3,
      "width": 100.2,
      "height": 50.8,
      "rotation": 0.785,
      "confidence": 0.85,
      "class_id": 0,
      "class_name": "label"
    }
  ],
  "timing": {
    "total_ms": 45.2,
    "inference_ms": 32.1
  }
}</pre>
            
            <h4>Client Libraries:</h4>
            <p>Use the provided Python client or connect with any WebSocket client that can send base64 image data.</p>
        </div>
        
        <script>
            // Auto-refresh connection count every 5 seconds
            setInterval(() => {
                fetch('/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('connections').textContent = data.active_connections;
                    })
                    .catch(error => console.log('Stats update failed:', error));
            }, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    avg_inference = 0
    if yolo_server.inference_times:
        avg_inference = round(np.mean(yolo_server.inference_times) * 1000, 2)
    
    return {
        "active_connections": len(yolo_server.active_connections),
        "total_requests": yolo_server.total_requests,
        "avg_inference_time_ms": avg_inference,
        "device": yolo_server.device,
        "model_path": yolo_server.model_path
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": yolo_server.model is not None,
        "device": yolo_server.device
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for YOLO inference"""
    await yolo_server.connect(websocket)
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_text()
            
            # Check if it's a ping/status request
            if data.strip() == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time(),
                    "active_connections": len(yolo_server.active_connections)
                }))
                continue
            
            # Check if it's a stats request
            if data.strip() == "stats":
                stats = await get_stats()
                await websocket.send_text(json.dumps({
                    "type": "stats",
                    **stats
                }))
                continue
            
            # Process as image data
            await yolo_server.process_image(websocket, data)
            
    except WebSocketDisconnect:
        yolo_server.disconnect(websocket)
        logger.info("WebSocket connection closed normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        yolo_server.disconnect(websocket)

def main():
    """Run the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO WebSocket Server")
    parser.add_argument("--model", default="runs/obb/train4/weights/best.pt", 
                       help="Path to YOLO model file")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, 
                       help="Port to bind to")
    parser.add_argument("--reload", action="store_true", 
                       help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Update global server instance with custom model path
    global yolo_server
    if args.model != "runs/obb/train4/weights/best.pt":
        yolo_server = YOLOWebSocketServer(args.model)
    
    logger.info(f"Starting YOLO WebSocket Server on {args.host}:{args.port}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {yolo_server.device}")
    
    uvicorn.run(
        "improved_ws_yolo:app" if not args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()