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
from typing import List, Dict, Any, Optional
import easyocr
from PIL import Image, ImageDraw, ImageFont
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOOCRPipeline:
    def __init__(self, 
                 yolo_model_path: str = "runs/obb/train4/weights/best.pt",
                 ocr_languages: List[str] = ['en']):
        self.yolo_model_path = yolo_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = None
        self.ocr_reader = None
        self.ocr_languages = ocr_languages
        self.active_connections: List[WebSocket] = []
        
        # Performance tracking
        self.inference_times = []
        self.ocr_times = []
        self.total_requests = 0
        
        # OCR settings
        self.min_confidence = 0.3
        self.enable_ocr = True
        self.ocr_padding = 10  # Pixels to pad around extracted regions
        
        self.load_models()
        
    def load_models(self):
        """Load YOLO and OCR models"""
        try:
            # Load YOLO model
            logger.info(f"Loading YOLO model from {self.yolo_model_path}")
            logger.info(f"Using device: {self.device}")
            
            self.yolo_model = ultralytics.YOLO(self.yolo_model_path).to(self.device)
            logger.info("YOLO model loaded successfully")
            
            if hasattr(self.yolo_model, 'names'):
                logger.info(f"YOLO model classes: {self.yolo_model.names}")
            
            # Load OCR model
            logger.info(f"Loading EasyOCR model for languages: {self.ocr_languages}")
            self.ocr_reader = easyocr.Reader(self.ocr_languages, gpu=torch.cuda.is_available())
            logger.info("OCR model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
            
    def extract_rotated_region(self, image: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """Extract and rotate a region from the image based on OBB detection"""
        try:
            center_x = detection['center_x']
            center_y = detection['center_y']
            width = detection['width']
            height = detection['height']
            rotation = detection['rotation']  # in radians
            
            # Add padding
            padded_width = width + 2 * self.ocr_padding
            padded_height = height + 2 * self.ocr_padding
            
            # Create rotation matrix
            cos_r = np.cos(-rotation)  # Negative to rotate back to horizontal
            sin_r = np.sin(-rotation)
            
            # Calculate the size of the rotated bounding box
            abs_cos = abs(cos_r)
            abs_sin = abs(sin_r)
            bound_width = int(padded_height * abs_sin + padded_width * abs_cos)
            bound_height = int(padded_height * abs_cos + padded_width * abs_sin)
            
            # Create transformation matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(rotation), 1.0)
            
            # Adjust the transformation matrix to include translation
            M[0, 2] += bound_width / 2 - center_x
            M[1, 2] += bound_height / 2 - center_y
            
            # Apply rotation to the entire image
            rotated_image = cv2.warpAffine(image, M, (bound_width, bound_height), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
            
            # Extract the region of interest from the rotated image
            start_x = max(0, bound_width // 2 - int(padded_width // 2))
            start_y = max(0, bound_height // 2 - int(padded_height // 2))
            end_x = min(bound_width, start_x + int(padded_width))
            end_y = min(bound_height, start_y + int(padded_height))
            
            extracted_region = rotated_image[start_y:end_y, start_x:end_x]
            
            # Ensure the extracted region is not empty
            if extracted_region.size == 0:
                logger.warning("Extracted region is empty")
                return None
                
            return extracted_region
            
        except Exception as e:
            logger.error(f"Error extracting rotated region: {e}")
            return None
            
    def perform_ocr(self, image_region: np.ndarray) -> List[Dict]:
        """Perform OCR on an image region"""
        try:
            if image_region is None or image_region.size == 0:
                return []
                
            # Ensure image is in the right format
            if len(image_region.shape) == 3:
                # Convert BGR to RGB for EasyOCR
                image_rgb = cv2.cvtColor(image_region, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image_region
                
            # Perform OCR
            start_time = time.time()
            results = self.ocr_reader.readtext(image_rgb, detail=1)
            ocr_time = time.time() - start_time
            
            self.ocr_times.append(ocr_time)
            if len(self.ocr_times) > 100:
                self.ocr_times.pop(0)
            
            # Process results
            ocr_results = []
            for (bbox, text, confidence) in results:
                if confidence >= self.min_confidence:
                    ocr_results.append({
                        "text": text,
                        "confidence": float(confidence),
                        "bbox": [[float(x), float(y)] for x, y in bbox]
                    })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return []
            
    def create_annotated_image(self, image: np.ndarray, detections: List[Dict], 
                             ocr_results: Dict[int, List[Dict]]) -> np.ndarray:
        """Create annotated image with both YOLO detections and OCR results"""
        annotated = image.copy()
        
        for i, detection in enumerate(detections):
            try:
                # Draw YOLO OBB
                center_x = detection['center_x']
                center_y = detection['center_y']
                width = detection['width']
                height = detection['height']
                rotation = detection['rotation']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Calculate rotated rectangle points
                cos_r = np.cos(rotation)
                sin_r = np.sin(rotation)
                
                # Half dimensions
                hw, hh = width / 2, height / 2
                
                # Corner points relative to center
                corners = np.array([
                    [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
                ])
                
                # Rotate and translate
                rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
                rotated_corners = corners @ rotation_matrix.T
                final_corners = rotated_corners + [center_x, center_y]
                
                # Convert to integer points
                points = final_corners.astype(int)
                
                # Draw rotated rectangle
                cv2.polylines(annotated, [points], True, (0, 255, 0), 2)
                
                # Draw YOLO label
                yolo_label = f'{class_name}: {confidence:.2f}'
                label_pos = (int(center_x - width/4), int(center_y - height/2 - 30))
                
                # Draw label background
                (label_width, label_height), _ = cv2.getTextSize(yolo_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, 
                            (label_pos[0], label_pos[1] - label_height - 5),
                            (label_pos[0] + label_width, label_pos[1] + 5),
                            (0, 255, 0), -1)
                
                cv2.putText(annotated, yolo_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw OCR results if available
                if i in ocr_results and ocr_results[i]:
                    ocr_y_offset = 15
                    for j, ocr_result in enumerate(ocr_results[i]):
                        ocr_text = ocr_result['text']
                        ocr_conf = ocr_result['confidence']
                        
                        # Truncate long text
                        if len(ocr_text) > 20:
                            display_text = ocr_text[:17] + "..."
                        else:
                            display_text = ocr_text
                            
                        ocr_label = f'OCR: "{display_text}" ({ocr_conf:.2f})'
                        ocr_pos = (int(center_x - width/4), int(center_y - height/2 - 10 + ocr_y_offset))
                        
                        # Draw OCR label background
                        (ocr_w, ocr_h), _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated,
                                    (ocr_pos[0], ocr_pos[1] - ocr_h - 3),
                                    (ocr_pos[0] + ocr_w, ocr_pos[1] + 3),
                                    (255, 255, 0), -1)
                        
                        cv2.putText(annotated, ocr_label, ocr_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        ocr_y_offset += 20
                        
            except Exception as e:
                logger.error(f"Error annotating detection {i}: {e}")
                
        return annotated
        
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        welcome_msg = {
            "type": "info",
            "message": "Connected to YOLO+OCR Pipeline Server",
            "device": self.device,
            "yolo_model": self.yolo_model_path,
            "ocr_languages": self.ocr_languages,
            "ocr_enabled": self.enable_ocr,
            "total_requests_processed": self.total_requests
        }
        try:
            await websocket.send_text(json.dumps(welcome_msg))
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")
        
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
        
    async def process_image(self, websocket: WebSocket, image_data: str):
        """Process image through YOLO+OCR pipeline"""
        try:
            total_start_time = time.time()
            
            # Decode image
            try:
                img_data = base64.b64decode(image_data)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_error:
                logger.error(f"Image decode error: {decode_error}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Failed to decode image: {str(decode_error)}"
                }))
                return
            
            if img is None:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Failed to decode image - invalid format"
                }))
                return
                
            # Run YOLO inference
            yolo_start = time.time()
            try:
                results = self.yolo_model.predict(img, imgsz=640, verbose=False)
            except Exception as yolo_error:
                logger.error(f"YOLO inference error: {yolo_error}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"YOLO inference failed: {str(yolo_error)}"
                }))
                return
            yolo_end = time.time()
            
            # Extract YOLO detections
            try:
                detection_data = self.extract_yolo_detections(results)
                detections = detection_data["obb"]
            except Exception as extract_error:
                logger.error(f"YOLO extraction error: {extract_error}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": f"Detection extraction failed: {str(extract_error)}"
                }))
                return
            
            # Perform OCR on detected regions
            ocr_results = {}
            ocr_start = time.time()
            
            if self.enable_ocr and detections:
                for i, detection in enumerate(detections):
                    try:
                        # Extract rotated region
                        extracted_region = self.extract_rotated_region(img, detection)
                        
                        if extracted_region is not None:
                            # Perform OCR
                            ocr_result = self.perform_ocr(extracted_region)
                            if ocr_result:
                                ocr_results[i] = ocr_result
                                
                    except Exception as ocr_error:
                        logger.warning(f"OCR failed for detection {i}: {ocr_error}")
                        
            ocr_end = time.time()
            
            # Create annotated image
            try:
                annotated_img = self.create_annotated_image(img, detections, ocr_results)
                
                # Encode annotated image
                _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as annotation_error:
                logger.error(f"Annotation error: {annotation_error}")
                annotated_b64 = None
            
            total_end = time.time()
            
            # Calculate timing
            total_time = total_end - total_start_time
            yolo_time = yolo_end - yolo_start
            ocr_time = ocr_end - ocr_start
            
            # Track performance
            self.inference_times.append(yolo_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            self.total_requests += 1
            
            # Prepare response
            response = {
                "type": "pipeline_result",
                "yolo_detections": detections,
                "ocr_results": ocr_results,
                "annotated_image": annotated_b64,
                "timing": {
                    "total_ms": round(total_time * 1000, 2),
                    "yolo_ms": round(yolo_time * 1000, 2),
                    "ocr_ms": round(ocr_time * 1000, 2)
                },
                "image_info": {
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "channels": int(img.shape[2])
                },
                "performance": {
                    "avg_yolo_ms": round(np.mean(self.inference_times) * 1000, 2) if self.inference_times else 0,
                    "avg_ocr_ms": round(np.mean(self.ocr_times) * 1000, 2) if self.ocr_times else 0,
                    "total_requests": self.total_requests
                },
                "settings": {
                    "ocr_enabled": self.enable_ocr,
                    "ocr_languages": self.ocr_languages,
                    "min_ocr_confidence": self.min_confidence
                }
            }
            
            # Send results
            try:
                await websocket.send_text(json.dumps(response))
                logger.debug(f"Processed pipeline: {len(detections)} detections, {sum(len(v) for v in ocr_results.values())} OCR results")
            except Exception as send_error:
                logger.error(f"Failed to send response: {send_error}")
                
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Pipeline processing failed: {str(e)}"
                }))
            except:
                logger.error("Failed to send error message to client")
                
    def extract_yolo_detections(self, results) -> Dict[str, Any]:
        """Extract detection data from YOLO results"""
        detection_data = {"obb": [], "boxes": []}
        
        try:
            for r in results:
                if hasattr(r, 'obb') and r.obb is not None:
                    try:
                        if hasattr(r.obb, 'xywhr') and len(r.obb.xywhr) > 0:
                            xywhr = r.obb.xywhr.cpu().numpy()
                            confs = r.obb.conf.cpu().numpy() if hasattr(r.obb, 'conf') else np.ones(len(xywhr)) * 0.8
                            
                            if hasattr(r.obb, 'cls') and r.obb.cls is not None:
                                classes = r.obb.cls.cpu().numpy()
                            else:
                                classes = np.zeros(len(xywhr))
                                
                            min_len = min(len(xywhr), len(confs), len(classes))
                            
                            for i in range(min_len):
                                box = xywhr[i]
                                conf = float(confs[i])
                                cls = int(classes[i])
                                
                                detection_data["obb"].append({
                                    "center_x": float(box[0]),
                                    "center_y": float(box[1]),
                                    "width": float(box[2]),
                                    "height": float(box[3]),
                                    "rotation": float(box[4]),
                                    "confidence": conf,
                                    "class_id": cls,
                                    "class_name": self.yolo_model.names.get(cls, f"Class_{cls}") if hasattr(self.yolo_model, 'names') else f"Class_{cls}"
                                })
                    except Exception as obb_error:
                        logger.warning(f"Error extracting OBB data: {obb_error}")
                        
        except Exception as e:
            logger.error(f"Error in extract_yolo_detections: {e}")
        
        return detection_data

# Create FastAPI app
app = FastAPI(title="YOLO+OCR Pipeline Server", version="1.0.0")

# Initialize pipeline
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = YOLOOCRPipeline()
    return pipeline

@app.get("/")
async def get():
    """Serve a simple test page"""
    server = get_pipeline()
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO+OCR Pipeline Server</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .success {{ background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .info {{ background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>YOLO+OCR Pipeline Server</h1>
            
            <div class="status success">
                ✅ Pipeline server is running
            </div>
            
            <div class="info">
                <h3>Pipeline Information:</h3>
                <ul>
                    <li><strong>WebSocket URL:</strong> ws://localhost:8000/ws</li>
                    <li><strong>Device:</strong> {server.device}</li>
                    <li><strong>YOLO Model:</strong> {server.yolo_model_path}</li>
                    <li><strong>OCR Languages:</strong> {', '.join(server.ocr_languages)}</li>
                    <li><strong>OCR Enabled:</strong> {server.enable_ocr}</li>
                    <li><strong>Total Requests:</strong> {server.total_requests}</li>
                </ul>
            </div>
            
            <h3>Features:</h3>
            <ul>
                <li>YOLO v8 Oriented Bounding Box Detection</li>
                <li>Automatic text region extraction and rotation</li>
                <li>EasyOCR text recognition</li>
                <li>Combined annotations</li>
                <li>Real-time performance metrics</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/settings")
async def get_settings():
    """Get current pipeline settings"""
    server = get_pipeline()
    return {
        "ocr_enabled": server.enable_ocr,
        "ocr_languages": server.ocr_languages,
        "min_confidence": server.min_confidence,
        "ocr_padding": server.ocr_padding,
        "device": server.device
    }

@app.post("/settings")
async def update_settings(settings: dict):
    """Update pipeline settings"""
    server = get_pipeline()
    
    if "ocr_enabled" in settings:
        server.enable_ocr = bool(settings["ocr_enabled"])
    if "min_confidence" in settings:
        server.min_confidence = float(settings["min_confidence"])
    if "ocr_padding" in settings:
        server.ocr_padding = int(settings["ocr_padding"])
        
    return {"status": "updated", "settings": await get_settings()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for pipeline processing"""
    server = get_pipeline()
    await server.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data.strip() == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time()
                }))
                continue
                
            # Process as image data
            await server.process_image(websocket, data)
            
    except WebSocketDisconnect:
        server.disconnect(websocket)
        logger.info("WebSocket connection closed normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        server.disconnect(websocket)

def main():
    """Run the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO+OCR Pipeline Server")
    parser.add_argument("--yolo-model", default="runs/obb/train4/weights/best.pt",
                       help="Path to YOLO model file")
    parser.add_argument("--ocr-languages", nargs='+', default=['en'],
                       help="OCR languages (e.g., en zh)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    global pipeline
    pipeline = YOLOOCRPipeline(args.yolo_model, args.ocr_languages)
    
    logger.info(f"Starting YOLO+OCR Pipeline Server on {args.host}:{args.port}")
    logger.info(f"YOLO Model: {args.yolo_model}")
    logger.info(f"OCR Languages: {args.ocr_languages}")
    logger.info(f"Device: {pipeline.device}")
    
    uvicorn.run("yolo_ocr_pipeline:app", host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()