# filename: enhanced_yolo_ocr_pipeline.py
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
from typing import List, Dict, Any, Optional, Tuple
import easyocr
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
import math
import pytesseract
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedImageProcessor:
    """Advanced image preprocessing for better OCR results"""
    
    def __init__(self):
        self.debug_mode = False
        
    def find_label_edges(self, image: np.ndarray, obb_info: Dict) -> Optional[np.ndarray]:
        """
        Find the actual edges of a label within the OBB detection using edge detection
        and contour analysis to exclude background/container areas
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Filter contours by area and aspect ratio
            min_area = (image.shape[0] * image.shape[1]) * 0.1  # At least 10% of the image
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Check if contour is roughly rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # At least quadrilateral
                        valid_contours.append((contour, area))
            
            if not valid_contours:
                return None
            
            # Get the largest valid contour (likely the label boundary)
            largest_contour = max(valid_contours, key=lambda x: x[1])[0]
            
            # Get the bounding rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            
            return box
            
        except Exception as e:
            logger.warning(f"Edge detection failed: {e}")
            return None
    
    def extract_and_rectify_label(self, image: np.ndarray, detection: Dict) -> Optional[np.ndarray]:
        """
        Extract label region using OBB and apply perspective correction
        """
        try:
            center_x = detection['center_x']
            center_y = detection['center_y']
            width = detection['width']
            height = detection['height']
            rotation = detection['rotation']
            
            # Calculate OBB corners
            cos_r = np.cos(rotation)
            sin_r = np.sin(rotation)
            
            # Half dimensions with padding
            padding = 20
            hw = (width + padding) / 2
            hh = (height + padding) / 2
            
            # Corner points relative to center
            corners = np.array([
                [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
            ])
            
            # Rotate and translate
            rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            rotated_corners = corners @ rotation_matrix.T
            final_corners = rotated_corners + [center_x, center_y]
            
            # Extract the region
            src_points = final_corners.astype(np.float32)
            
            # Define destination rectangle (rectified)
            dst_width = int(width + padding)
            dst_height = int(height + padding)
            dst_points = np.array([
                [0, 0], [dst_width, 0], [dst_width, dst_height], [0, dst_height]
            ], dtype=np.float32)
            
            # Get perspective transform matrix
            M = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transform
            rectified = cv2.warpPerspective(image, M, (dst_width, dst_height))
            
            # Try to find refined label edges within the rectified region
            label_edges = self.find_label_edges(rectified, detection)
            
            if label_edges is not None:
                # Extract only the label area
                label_rect = cv2.minAreaRect(label_edges)
                label_box = cv2.boxPoints(label_rect)
                label_box = np.float32(label_box)
                
                # Create tighter crop
                label_width = int(label_rect[1][0])
                label_height = int(label_rect[1][1])
                
                if label_width > 0 and label_height > 0:
                    label_dst = np.array([
                        [0, 0], [label_width, 0], 
                        [label_width, label_height], [0, label_height]
                    ], dtype=np.float32)
                    
                    label_M = cv2.getPerspectiveTransform(label_box, label_dst)
                    final_label = cv2.warpPerspective(rectified, label_M, (label_width, label_height))
                    
                    return final_label
            
            return rectified
            
        except Exception as e:
            logger.warning(f"Label rectification failed: {e}")
            return None
    
    def enhance_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive image preprocessing for optimal OCR
        """
        try:
            # Convert to PIL for advanced processing
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Resize if too small (OCR works better on larger images)
            width, height = pil_image.size
            if width < 300 or height < 100:
                scale_factor = max(300 / width, 100 / height, 2.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to grayscale for processing
            gray_pil = pil_image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(gray_pil)
            gray_pil = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(gray_pil)
            gray_pil = enhancer.enhance(1.2)
            
            # Apply noise reduction
            gray_pil = gray_pil.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert back to OpenCV format
            enhanced = np.array(gray_pil)
            
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Apply additional denoising
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            # Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_rotation_correction(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct text rotation using Hough line transform
        """
        try:
            # Apply edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Apply Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            
            if lines is not None:
                # Calculate angles of detected lines
                angles = []
                for line in lines[:20]:  # Consider only first 20 lines
                    rho, theta = line[0]  # Extract rho and theta from [[rho, theta]]
                    angle = theta * 180 / np.pi
                    # Convert to -90 to 90 degree range
                    if angle > 90:
                        angle -= 180
                    angles.append(angle)
                
                if angles:
                    # Find the most common angle (likely text orientation)
                    angle_bins = np.histogram(angles, bins=36, range=(-90, 90))[0]
                    dominant_angle_idx = np.argmax(angle_bins)
                    dominant_angle = (dominant_angle_idx * 5) - 90
                    
                    # Only correct if angle is significant
                    if abs(dominant_angle) > 2:
                        center = (image.shape[1] // 2, image.shape[0] // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)
                        
                        # Calculate new image dimensions
                        cos_val = np.abs(rotation_matrix[0, 0])
                        sin_val = np.abs(rotation_matrix[0, 1])
                        new_width = int((image.shape[0] * sin_val) + (image.shape[1] * cos_val))
                        new_height = int((image.shape[0] * cos_val) + (image.shape[1] * sin_val))
                        
                        # Adjust rotation matrix for new dimensions
                        rotation_matrix[0, 2] += (new_width / 2) - center[0]
                        rotation_matrix[1, 2] += (new_height / 2) - center[1]
                        
                        corrected = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        
                        return corrected, dominant_angle
            
            return image, 0.0
            
        except Exception as e:
            logger.warning(f"Rotation correction failed: {e}")
            return image, 0.0

class RobustOCREngine:
    """
    Multi-engine OCR system with confidence-based selection
    """
    
    def __init__(self, languages: List[str] = ['en']):
        self.languages = languages
        self.easyocr_reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
        
        # Configure Tesseract if available
        self.tesseract_available = False
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
            logger.info("Tesseract OCR available")
        except:
            logger.warning("Tesseract OCR not available")
    
    def easyocr_extract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using EasyOCR"""
        try:
            results = self.easyocr_reader.readtext(image, detail=1, paragraph=True)
            
            ocr_results = []
            for result in results:
                if len(result) != 3:
                    logger.warning(f"Unexpected EasyOCR result format: {result}")
                    continue
                bbox, text, confidence = result
                if confidence > 0.1:  # Lower threshold for initial filtering
                    ocr_results.append({
                        "text": text.strip(),
                        "confidence": float(confidence),
                        "bbox": [[float(x), float(y)] for x, y in bbox],
                        "engine": "easyocr"
                    })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def tesseract_extract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using Tesseract OCR"""
        if not self.tesseract_available:
            return []
            
        try:
            # Configure Tesseract
            config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/() '
            
            # Get detailed results
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            ocr_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                confidence = float(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence > 30 and len(text) > 0:  # Tesseract confidence is 0-100
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    
                    ocr_results.append({
                        "text": text,
                        "confidence": confidence / 100.0,  # Normalize to 0-1
                        "bbox": bbox,
                        "engine": "tesseract"
                    })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def combined_extract(self, image: np.ndarray) -> List[Dict]:
        """
        Run multiple OCR engines and combine results intelligently
        """
        all_results = []
        
        # Run EasyOCR
        easyocr_results = self.easyocr_extract(image)
        all_results.extend(easyocr_results)
        
        # Run Tesseract if available
        if self.tesseract_available:
            tesseract_results = self.tesseract_extract(image)
            all_results.extend(tesseract_results)
        
        if not all_results:
            return []
        
        # Merge overlapping results and select best confidence
        merged_results = self.merge_overlapping_results(all_results)
        
        # Filter by confidence and length
        filtered_results = []
        for result in merged_results:
            text = result["text"].strip()
            if len(text) >= 2 and result["confidence"] > 0.3:
                filtered_results.append(result)
        
        # Sort by confidence
        filtered_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return filtered_results
    
    def merge_overlapping_results(self, results: List[Dict]) -> List[Dict]:
        """
        Merge OCR results from different engines that detect the same text
        """
        if len(results) <= 1:
            return results
        
        def bbox_overlap(bbox1, bbox2):
            """Calculate IoU of two bounding boxes"""
            try:
                # Convert to format [x1, y1, x2, y2]
                box1 = [min(p[0] for p in bbox1), min(p[1] for p in bbox1),
                       max(p[0] for p in bbox1), max(p[1] for p in bbox1)]
                box2 = [min(p[0] for p in bbox2), min(p[1] for p in bbox2),
                       max(p[0] for p in bbox2), max(p[1] for p in bbox2)]
                
                # Calculate intersection
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                
                if x2 <= x1 or y2 <= y1:
                    return 0.0
                
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection
                
                return intersection / union if union > 0 else 0.0
            except:
                return 0.0
        
        merged = []
        used = set()
        
        for i, result1 in enumerate(results):
            if i in used:
                continue
                
            best_result = result1
            used.add(i)
            
            # Look for overlapping results
            for j, result2 in enumerate(results[i+1:], i+1):
                if j in used:
                    continue
                
                # Check bbox overlap
                overlap = bbox_overlap(result1["bbox"], result2["bbox"])
                
                # Check text similarity
                text_similarity = self.text_similarity(result1["text"], result2["text"])
                
                if overlap > 0.5 or text_similarity > 0.8:
                    # Choose the result with higher confidence
                    if result2["confidence"] > best_result["confidence"]:
                        best_result = result2
                    used.add(j)
            
            merged.append(best_result)
        
        return merged
    
    def text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using character overlap"""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0

class EnhancedYOLOOCRPipeline:
    def __init__(self, 
                 yolo_model_path: str = "runs/obb/train4/weights/best.pt",
                 ocr_languages: List[str] = ['en']):
        self.yolo_model_path = yolo_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_model = None
        self.image_processor = AdvancedImageProcessor()
        self.ocr_engine = RobustOCREngine(ocr_languages)
        self.active_connections: List[WebSocket] = []
        
        # Performance tracking
        self.inference_times = []
        self.ocr_times = []
        self.preprocessing_times = []
        self.total_requests = 0
        
        # Processing settings
        self.min_confidence = 0.3
        self.enable_ocr = True
        self.enable_preprocessing = True
        self.enable_rotation_correction = True
        self.debug_mode = False
        
        self.load_models()
        
    def load_models(self):
        """Load YOLO model"""
        try:
            logger.info(f"Loading YOLO model from {self.yolo_model_path}")
            logger.info(f"Using device: {self.device}")
            
            self.yolo_model = ultralytics.YOLO(self.yolo_model_path).to(self.device)
            logger.info("YOLO model loaded successfully")
            
            if hasattr(self.yolo_model, 'names'):
                logger.info(f"YOLO model classes: {self.yolo_model.names}")
            
            logger.info("OCR engines initialized")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def process_detection_advanced(self, image: np.ndarray, detection: Dict) -> List[Dict]:
        try:
            preprocess_start = time.time()
            
            # Step 1: Extract and rectify label region
            extracted_region = self.image_processor.extract_and_rectify_label(image, detection)
            
            if extracted_region is None or extracted_region.size == 0:
                logger.warning(f"Invalid extracted region for detection: {detection}")
                return []
            
            # Validate image dimensions
            if extracted_region.shape[0] < 10 or extracted_region.shape[1] < 10:
                logger.warning(f"Extracted region too small: {extracted_region.shape}")
                return []
            
            # Step 2: Enhance image for OCR
            if self.enable_preprocessing:
                enhanced_region = self.image_processor.enhance_image_for_ocr(extracted_region)
                
                # Step 3: Apply rotation correction
                if self.enable_rotation_correction:
                    enhanced_region, rotation_angle = self.image_processor.apply_rotation_correction(enhanced_region)
                    if abs(rotation_angle) > 2:
                        logger.debug(f"Applied rotation correction: {rotation_angle:.1f} degrees")
            else:
                enhanced_region = extracted_region
            
            preprocess_end = time.time()
            self.preprocessing_times.append(preprocess_end - preprocess_start)
            if len(self.preprocessing_times) > 100:
                self.preprocessing_times.pop(0)
            
            # Step 4: Perform robust OCR
            ocr_start = time.time()
            ocr_results = self.ocr_engine.combined_extract(enhanced_region)
            ocr_end = time.time()
            
            self.ocr_times.append(ocr_end - ocr_start)
            if len(self.ocr_times) > 100:
                self.ocr_times.pop(0)
            
            # Step 5: Post-process results
            processed_results = self.post_process_ocr_results(ocr_results)
            
            # Add debug information if enabled
            if self.debug_mode:
                for result in processed_results:
                    result["debug_info"] = {
                        "preprocessing_time": preprocess_end - preprocess_start,
                        "original_size": f"{extracted_region.shape[1]}x{extracted_region.shape[0]}",
                        "enhanced_size": f"{enhanced_region.shape[1]}x{enhanced_region.shape[0]}"
                    }
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Advanced processing failed: {e}")
            return []
    
    def post_process_ocr_results(self, ocr_results: List[Dict]) -> List[Dict]:
        """
        Clean up and validate OCR results
        """
        if not ocr_results:
            return []
        
        processed = []
        
        for result in ocr_results:
            text = result["text"].strip()
            
            # Basic text cleaning
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Skip if too short or low confidence
            if len(text) < 2 or result["confidence"] < self.min_confidence:
                continue
            
            # Skip if text contains mostly special characters
            alpha_ratio = sum(c.isalnum() for c in text) / len(text)
            if alpha_ratio < 0.3:
                continue
            
            result["text"] = text
            processed.append(result)
        
        return processed
    
    def create_debug_visualization(self, original_image: np.ndarray, 
                                 detections: List[Dict], 
                                 ocr_results: Dict[int, List[Dict]],
                                 processed_regions: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Create a debug visualization showing the processing steps
        """
        try:
            # Create a larger canvas for debug info
            debug_height = original_image.shape[0] + 400
            debug_width = max(original_image.shape[1], 1200)
            debug_canvas = np.ones((debug_height, debug_width, 3), dtype=np.uint8) * 255
            
            # Place original image
            debug_canvas[:original_image.shape[0], :original_image.shape[1]] = original_image
            
            # Draw detections on original
            annotated = self.create_annotated_image(original_image, detections, ocr_results)
            debug_canvas[:original_image.shape[0], :original_image.shape[1]] = annotated
            
            # Add processed regions below
            y_offset = original_image.shape[0] + 50
            x_offset = 10
            
            for i, (detection_idx, processed_region) in enumerate(processed_regions.items()):
                if processed_region is not None and i < 4:  # Show max 4 regions
                    # Resize region for display
                    display_height = 80
                    aspect_ratio = processed_region.shape[1] / processed_region.shape[0]
                    display_width = int(display_height * aspect_ratio)
                    
                    if len(processed_region.shape) == 2:
                        processed_region_color = cv2.cvtColor(processed_region, cv2.COLOR_GRAY2BGR)
                    else:
                        processed_region_color = processed_region
                    
                    resized_region = cv2.resize(processed_region_color, (display_width, display_height))
                    
                    # Place on debug canvas
                    if x_offset + display_width < debug_width:
                        debug_canvas[y_offset:y_offset+display_height, 
                                   x_offset:x_offset+display_width] = resized_region
                        
                        # Add label
                        label = f"Region {detection_idx}"
                        cv2.putText(debug_canvas, label, (x_offset, y_offset-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        x_offset += display_width + 20
            
            return debug_canvas
            
        except Exception as e:
            logger.error(f"Debug visualization failed: {e}")
            return original_image
    
    def create_annotated_image(self, image: np.ndarray, detections: List[Dict], 
                             ocr_results: Dict[int, List[Dict]]) -> np.ndarray:
        """Create annotated image with enhanced visualization"""
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
                
                # Draw rotated rectangle with thicker line
                cv2.polylines(annotated, [points], True, (0, 255, 0), 3)
                
                # Draw YOLO label with better background
                yolo_label = f'{class_name}: {confidence:.2f}'
                label_pos = (int(center_x - width/4), int(center_y - height/2 - 40))
                
                # Calculate label size
                (label_width, label_height), baseline = cv2.getTextSize(
                    yolo_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Draw label background
                cv2.rectangle(annotated, 
                            (label_pos[0] - 5, label_pos[1] - label_height - baseline - 5),
                            (label_pos[0] + label_width + 5, label_pos[1] + baseline + 5),
                            (0, 255, 0), -1)
                
                cv2.putText(annotated, yolo_label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Draw OCR results with improved formatting
                if i in ocr_results and ocr_results[i]:
                    ocr_y_offset = 20
                    for j, ocr_result in enumerate(ocr_results[i][:3]):  # Show max 3 OCR results
                        ocr_text = ocr_result['text']
                        ocr_conf = ocr_result['confidence']
                        engine = ocr_result.get('engine', 'unknown')
                        
                        # Truncate long text
                        if len(ocr_text) > 25:
                            display_text = ocr_text[:22] + "..."
                        else:
                            display_text = ocr_text
                            
                        ocr_label = f'[{engine.upper()}] "{display_text}" ({ocr_conf:.2f})'
                        ocr_pos = (int(center_x - width/4), int(center_y - height/2 - 15 + ocr_y_offset))
                        
                        # Calculate OCR label size
                        (ocr_w, ocr_h), ocr_baseline = cv2.getTextSize(
                            ocr_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        # Choose color based on engine
                        bg_color = (255, 255, 0) if engine == 'easyocr' else (0, 255, 255)
                        
                        # Draw OCR label background
                        cv2.rectangle(annotated,
                                    (ocr_pos[0] - 3, ocr_pos[1] - ocr_h - ocr_baseline - 3),
                                    (ocr_pos[0] + ocr_w + 3, ocr_pos[1] + ocr_baseline + 3),
                                    bg_color, -1)
                        
                        cv2.putText(annotated, ocr_label, ocr_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        ocr_y_offset += 25
                        
            except Exception as e:
                logger.error(f"Error annotating detection {i}: {e}")
                
        return annotated
    
    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message with enhanced info
        welcome_msg = {
            "type": "info",
            "message": "Connected to Enhanced YOLO+OCR Pipeline Server",
            "device": self.device,
            "yolo_model": self.yolo_model_path,
            "ocr_languages": self.ocr_engine.languages,
            "ocr_enabled": self.enable_ocr,
            "preprocessing_enabled": self.enable_preprocessing,
            "rotation_correction_enabled": self.enable_rotation_correction,
            "tesseract_available": self.ocr_engine.tesseract_available,
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
        """Process image through enhanced YOLO+OCR pipeline"""
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
            
            # Perform enhanced OCR processing
            ocr_results = {}
            processed_regions = {}
            ocr_start = time.time()
            
            if self.enable_ocr and detections:
                for i, detection in enumerate(detections):
                    try:
                        # Use advanced processing pipeline
                        ocr_result = self.process_detection_advanced(img, detection)
                        
                        if ocr_result:
                            ocr_results[i] = ocr_result
                            
                        # Store processed region for debugging if enabled
                        if self.debug_mode:
                            extracted_region = self.image_processor.extract_and_rectify_label(img, detection)
                            if extracted_region is not None:
                                processed_regions[i] = extracted_region
                                
                    except Exception as ocr_error:
                        logger.warning(f"Enhanced OCR failed for detection {i}: {ocr_error}")
                        
            ocr_end = time.time()
            
            # Create visualization
            try:
                if self.debug_mode and processed_regions:
                    annotated_img = self.create_debug_visualization(img, detections, ocr_results, processed_regions)
                else:
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
            avg_preprocessing_time = np.mean(self.preprocessing_times) if self.preprocessing_times else 0
            
            # Track performance
            self.inference_times.append(yolo_time)
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            self.total_requests += 1
            
            # Count total OCR results
            total_ocr_count = sum(len(results) for results in ocr_results.values())
            
            # Prepare enhanced response
            response = {
                "type": "pipeline_result",
                "yolo_detections": detections,
                "ocr_results": ocr_results,
                "annotated_image": annotated_b64,
                "timing": {
                    "total_ms": round(total_time * 1000, 2),
                    "yolo_ms": round(yolo_time * 1000, 2),
                    "ocr_ms": round(ocr_time * 1000, 2),
                    "preprocessing_ms": round(avg_preprocessing_time * 1000, 2)
                },
                "image_info": {
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "channels": int(img.shape[2])
                },
                "processing_info": {
                    "detections_found": len(detections),
                    "total_ocr_results": total_ocr_count,
                    "preprocessing_enabled": self.enable_preprocessing,
                    "rotation_correction_enabled": self.enable_rotation_correction,
                    "engines_used": ["easyocr"] + (["tesseract"] if self.ocr_engine.tesseract_available else [])
                },
                "performance": {
                    "avg_yolo_ms": round(np.mean(self.inference_times) * 1000, 2) if self.inference_times else 0,
                    "avg_ocr_ms": round(np.mean(self.ocr_times) * 1000, 2) if self.ocr_times else 0,
                    "avg_preprocessing_ms": round(avg_preprocessing_time * 1000, 2),
                    "total_requests": self.total_requests
                },
                "settings": {
                    "ocr_enabled": self.enable_ocr,
                    "preprocessing_enabled": self.enable_preprocessing,
                    "rotation_correction_enabled": self.enable_rotation_correction,
                    "ocr_languages": self.ocr_engine.languages,
                    "min_ocr_confidence": self.min_confidence,
                    "debug_mode": self.debug_mode
                }
            }
            
            # Send results
            try:
                await websocket.send_text(json.dumps(response))
                logger.info(f"Enhanced pipeline processed: {len(detections)} detections, {total_ocr_count} OCR results")
            except Exception as send_error:
                logger.error(f"Failed to send response: {send_error}")
                
        except Exception as e:
            logger.error(f"Enhanced pipeline processing error: {e}")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Enhanced pipeline processing failed: {str(e)}"
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
app = FastAPI(title="Enhanced YOLO+OCR Pipeline Server", version="2.0.0")

# Initialize pipeline
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = EnhancedYOLOOCRPipeline()
    return pipeline

@app.get("/")
async def get():
    """Serve enhanced server info page"""
    server = get_pipeline()
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced YOLO+OCR Pipeline Server</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .status {{ padding: 15px; margin: 15px 0; border-radius: 8px; }}
            .success {{ background-color: #d4edda; color: #155724; border-left: 4px solid #28a745; }}
            .info {{ background-color: #d1ecf1; color: #0c5460; border-left: 4px solid #17a2b8; }}
            .feature {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 3px solid #6c757d; }}
            h1 {{ color: #2c3e50; margin-bottom: 20px; }}
            h3 {{ color: #34495e; margin-top: 25px; }}
            ul {{ margin-left: 20px; }}
            li {{ margin: 8px 0; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Enhanced YOLO+OCR Pipeline Server</h1>
            
            <div class="status success">
                ✅ Enhanced pipeline server is running with advanced preprocessing
            </div>
            
            <div class="grid">
                <div class="info">
                    <h3>📊 Server Information:</h3>
                    <ul>
                        <li><strong>WebSocket URL:</strong> ws://localhost:8000/ws</li>
                        <li><strong>Device:</strong> {server.device}</li>
                        <li><strong>YOLO Model:</strong> {server.yolo_model_path}</li>
                        <li><strong>OCR Languages:</strong> {', '.join(server.ocr_engine.languages)}</li>
                        <li><strong>Tesseract Available:</strong> {'Yes' if server.ocr_engine.tesseract_available else 'No'}</li>
                        <li><strong>Total Requests:</strong> {server.total_requests}</li>
                    </ul>
                </div>
                
                <div class="info">
                    <h3>⚙️ Processing Settings:</h3>
                    <ul>
                        <li><strong>OCR Enabled:</strong> {'Yes' if server.enable_ocr else 'No'}</li>
                        <li><strong>Preprocessing:</strong> {'Yes' if server.enable_preprocessing else 'No'}</li>
                        <li><strong>Rotation Correction:</strong> {'Yes' if server.enable_rotation_correction else 'No'}</li>
                        <li><strong>Min Confidence:</strong> {server.min_confidence}</li>
                        <li><strong>Debug Mode:</strong> {'Yes' if server.debug_mode else 'No'}</li>
                    </ul>
                </div>
            </div>
            
            <h3>🔧 Enhanced Features:</h3>
            
            <div class="feature">
                <strong>🎯 Advanced Label Detection:</strong>
                <ul>
                    <li>Edge detection and contour analysis to find actual label boundaries</li>
                    <li>Perspective correction using OBB rotation data</li>
                    <li>Automatic cropping to exclude container/background areas</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>📸 Image Preprocessing Pipeline:</strong>
                <ul>
                    <li>Intelligent image scaling for optimal OCR performance</li>
                    <li>Contrast and sharpness enhancement</li>
                    <li>Noise reduction using bilateral filtering</li>
                    <li>CLAHE (Contrast Limited Adaptive Histogram Equalization)</li>
                    <li>Morphological operations for text cleanup</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>🔄 Rotation Correction:</strong>
                <ul>
                    <li>Hough line transform for text orientation detection</li>
                    <li>Automatic rotation correction for skewed text</li>
                    <li>Smart angle detection using dominant line analysis</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>🧠 Multi-Engine OCR:</strong>
                <ul>
                    <li>EasyOCR with paragraph detection</li>
                    <li>Tesseract OCR with custom configuration</li>
                    <li>Intelligent result merging and confidence-based selection</li>
                    <li>Text similarity analysis for duplicate removal</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>🎨 Enhanced Visualization:</strong>
                <ul>
                    <li>Color-coded annotations by OCR engine</li>
                    <li>Debug mode with processing step visualization</li>
                    <li>Confidence scores and engine identification</li>
                    <li>Performance timing breakdown</li>
                </ul>
            </div>
            
            <h3>📈 Performance Tracking:</h3>
            <ul>
                <li>YOLO inference timing</li>
                <li>OCR processing timing</li>
                <li>Image preprocessing timing</li>
                <li>End-to-end pipeline performance</li>
                <li>Running averages and statistics</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/settings")
async def get_settings():
    """Get current enhanced pipeline settings"""
    server = get_pipeline()
    return {
        "ocr_enabled": server.enable_ocr,
        "preprocessing_enabled": server.enable_preprocessing,
        "rotation_correction_enabled": server.enable_rotation_correction,
        "ocr_languages": server.ocr_engine.languages,
        "min_confidence": server.min_confidence,
        "debug_mode": server.debug_mode,
        "tesseract_available": server.ocr_engine.tesseract_available,
        "device": server.device
    }

@app.post("/settings")
async def update_settings(settings: dict):
    """Update enhanced pipeline settings"""
    server = get_pipeline()
    
    if "ocr_enabled" in settings:
        server.enable_ocr = bool(settings["ocr_enabled"])
    if "preprocessing_enabled" in settings:
        server.enable_preprocessing = bool(settings["preprocessing_enabled"])
    if "rotation_correction_enabled" in settings:
        server.enable_rotation_correction = bool(settings["rotation_correction_enabled"])
    if "min_confidence" in settings:
        server.min_confidence = float(settings["min_confidence"])
    if "debug_mode" in settings:
        server.debug_mode = bool(settings["debug_mode"])
        
    return {"status": "updated", "settings": await get_settings()}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for pipeline processing"""
    server = get_pipeline()
    await server.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data.strip() == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": time.time(),
                    "server_info": {
                        "preprocessing_enabled": server.enable_preprocessing,
                        "rotation_correction_enabled": server.enable_rotation_correction,
                        "debug_mode": server.debug_mode
                    }
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
    """Run the enhanced server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced YOLO+OCR Pipeline Server")
    parser.add_argument("--yolo-model", default="runs/obb/train4/weights/best.pt",
                       help="Path to YOLO model file")
    parser.add_argument("--ocr-languages", nargs='+', default=['en'],
                       help="OCR languages (e.g., en zh)")
    parser.add_argument("--host", default="192.168.1.99", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable image preprocessing")
    parser.add_argument("--no-rotation-correction", action="store_true", help="Disable rotation correction")
    
    args = parser.parse_args()
    
    global pipeline
    pipeline = EnhancedYOLOOCRPipeline(args.yolo_model, args.ocr_languages)
    
    # Apply command line settings
    pipeline.debug_mode = args.debug
    pipeline.enable_preprocessing = not args.no_preprocessing
    pipeline.enable_rotation_correction = not args.no_rotation_correction
    
    logger.info(f"Starting Enhanced YOLO+OCR Pipeline Server on {args.host}:{args.port}")
    logger.info(f"YOLO Model: {args.yolo_model}")
    logger.info(f"OCR Languages: {args.ocr_languages}")
    logger.info(f"Device: {pipeline.device}")
    logger.info(f"Tesseract Available: {pipeline.ocr_engine.tesseract_available}")
    logger.info(f"Preprocessing Enabled: {pipeline.enable_preprocessing}")
    logger.info(f"Rotation Correction: {pipeline.enable_rotation_correction}")
    logger.info(f"Debug Mode: {pipeline.debug_mode}")
    
    uvicorn.run("enhanced_yolo_ocr_pipeline:app", host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()