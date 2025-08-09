# filename: enhanced_yolo_ocr_pipeline_v2.py
import cv2
import numpy as np
import base64
import json
import torch
import uvicorn
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import ultralytics
import time
from typing import List, Dict, Any, Optional, Tuple
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
import math
import pytesseract
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Custom JSON encoder to handle numpy types and other non-serializable objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'tolist'):  # Handle other numpy-like objects
            return obj.tolist()
        return super().default(obj)

def ensure_json_serializable(obj):
    """Recursively convert numpy types and other non-serializable objects to JSON serializable types"""
    if isinstance(obj, dict):
        return {key: ensure_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'tolist'):  # Handle other numpy-like objects
        return obj.tolist()
    else:
        return obj

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStep:
    """Data class for storing preprocessing step information"""
    name: str
    image: np.ndarray
    description: str
    params: Dict[str, Any] = None

class PerspectiveCorrectionEngine:
    """Advanced perspective correction with linear transformations"""
    
    def __init__(self):
        self.debug_mode = False
        self.processing_steps: List[ProcessingStep] = []
        # Angle handling options. Positive angles are assumed CCW (OpenCV convention)
        self.rotation_sign: float = 1.0  # set to -1.0 if YOLO OBB angle appears flipped
        self.auto_sign_disambiguation: bool = True
    
    def set_rotation_sign(self, sign: float) -> None:
        """Configure rotation sign to match YOLO OBB convention (1.0 or -1.0)."""
        self.rotation_sign = 1.0 if sign >= 0 else -1.0
    
    def clear_steps(self):
        """Clear processing steps for new image"""
        self.processing_steps = []
    
    def add_step(self, name: str, image: np.ndarray, description: str, params: Dict = None):
        """Add a processing step for visualization"""
        self.processing_steps.append(ProcessingStep(name, image.copy(), description, params or {}))
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in top-left, top-right, bottom-right, bottom-left order"""
        # Initialize ordered points
        rect = np.zeros((4, 2), dtype="float32")
        
        # Calculate sums and differences
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        # Top-right point has smallest difference
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def normalize_angle(self, angle_rad: float) -> float:
        """Normalize angle to [-pi/2, pi/2) and apply configured sign."""
        angle = float(self.rotation_sign) * float(angle_rad)
        # Wrap to [-pi, pi)
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        # Map to [-pi/2, pi/2)
        if angle >= np.pi/2:
            angle -= np.pi
        if angle < -np.pi/2:
            angle += np.pi
        return angle
    
    def normalize_angle_with_sign(self, angle_rad: float, sign: float) -> float:
        """Normalize angle to [-pi/2, pi/2) using a specific sign override."""
        angle = float(sign) * float(angle_rad)
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        if angle >= np.pi/2:
            angle -= np.pi
        if angle < -np.pi/2:
            angle += np.pi
        return angle
    
    def obb_to_corners(self, center_x: float, center_y: float, width: float, height: float, angle_rad: float) -> np.ndarray:
        """Convert OBB (cx, cy, w, h, angle) to 4 corners (TL, TR, BR, BL)."""
        angle = self.normalize_angle(angle_rad)
        cos_r = float(np.cos(angle))
        sin_r = float(np.sin(angle))
        half_w = float(width) / 2.0
        half_h = float(height) / 2.0
        # Unrotated corners relative to center
        base = np.array([
            [-half_w, -half_h],  # TL in unrotated frame
            [ half_w, -half_h],  # TR
            [ half_w,  half_h],  # BR
            [-half_w,  half_h],  # BL
        ], dtype=np.float32)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        rotated = base @ rot.T
        translated = rotated + np.array([center_x, center_y], dtype=np.float32)
        # Ensure TL, TR, BR, BL ordering strictly
        ordered = self.order_points(translated.astype(np.float32))
        return ordered

    def obb_to_corners_with_sign(self, center_x: float, center_y: float, width: float, height: float, angle_rad: float, sign: float) -> np.ndarray:
        angle = self.normalize_angle_with_sign(angle_rad, sign)
        cos_r = float(np.cos(angle))
        sin_r = float(np.sin(angle))
        half_w = float(width) / 2.0
        half_h = float(height) / 2.0
        base = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h],
        ], dtype=np.float32)
        rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        rotated = base @ rot.T
        translated = rotated + np.array([center_x, center_y], dtype=np.float32)
        return self.order_points(translated.astype(np.float32))
    
    def calculate_perspective_transform(self, obb_info: Dict, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Calculate perspective transformation matrix using OBB parameters - simplified and more accurate
        """
        center_x = float(obb_info['center_x'])
        center_y = float(obb_info['center_y'])
        width = float(obb_info['width'])
        height = float(obb_info['height'])
        rotation = float(obb_info.get('rotation', 0.0))
        
        # Compute corners from OBB (always returns TL, TR, BR, BL)
        src_points = self.obb_to_corners(center_x, center_y, width, height, rotation)
        
        # Check if all corners are within image bounds (with margin)
        margin = 10
        h_img, w_img = image_shape
        all_in_bounds = True
        for corner in src_points:
            if (corner[0] < -margin or corner[0] > w_img + margin or
                corner[1] < -margin or corner[1] > h_img + margin):
                all_in_bounds = False
                break
        
        if not all_in_bounds:
            # Fallback to simple rectangular crop around center
            padding = 15
            x1 = max(0, int(center_x - width/2 - padding))
            y1 = max(0, int(center_y - height/2 - padding))
            x2 = min(w_img, int(center_x + width/2 + padding))
            y2 = min(h_img, int(center_y + height/2 + padding))
            src_points = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            dst_width = x2 - x1
            dst_height = y2 - y1
        else:
            # Use OBB size with a small padding
            dst_width = int(max(50, min(600, round(width + 20))))
            dst_height = int(max(30, min(400, round(height + 20))))
        
        # Ensure reasonable output size
        dst_width = max(50, min(int(dst_width), 600))
        dst_height = max(30, min(int(dst_height), 400))
        
        # Destination points (always a perfect rectangle)
        dst_points = np.array([
            [0, 0],                    # top-left
            [dst_width, 0],            # top-right
            [dst_width, dst_height],   # bottom-right
            [0, dst_height]            # bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        return transform_matrix, (dst_width, dst_height)

    def calculate_perspective_transform_with_sign(self, obb_info: Dict, image_shape: Tuple[int, int], sign: float) -> Tuple[np.ndarray, Tuple[int, int]]:
        center_x = float(obb_info['center_x'])
        center_y = float(obb_info['center_y'])
        width = float(obb_info['width'])
        height = float(obb_info['height'])
        rotation = float(obb_info.get('rotation', 0.0))
        src_points = self.obb_to_corners_with_sign(center_x, center_y, width, height, rotation, sign)
        margin = 10
        h_img, w_img = image_shape
        all_in_bounds = True
        for corner in src_points:
            if (corner[0] < -margin or corner[0] > w_img + margin or
                corner[1] < -margin or corner[1] > h_img + margin):
                all_in_bounds = False
                break
        if not all_in_bounds:
            padding = 15
            x1 = max(0, int(center_x - width/2 - padding))
            y1 = max(0, int(center_y - height/2 - padding))
            x2 = min(w_img, int(center_x + width/2 + padding))
            y2 = min(h_img, int(center_y + height/2 + padding))
            src_points = np.array([[x1, y1],[x2, y1],[x2, y2],[x1, y2]], dtype=np.float32)
            dst_width = x2 - x1
            dst_height = y2 - y1
        else:
            dst_width = int(max(50, min(600, round(width + 20))))
            dst_height = int(max(30, min(400, round(height + 20))))
        dst_width = max(50, min(int(dst_width), 600))
        dst_height = max(30, min(int(dst_height), 400))
        dst_points = np.array([[0,0],[dst_width,0],[dst_width,dst_height],[0,dst_height]], dtype=np.float32)
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return transform_matrix, (dst_width, dst_height)

    def _orientation_score_deg(self, img_gray: np.ndarray) -> float:
        """Return absolute dominant line angle in degrees (lower is better horizontalness)."""
        try:
            edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            if lines is None:
                return 90.0
            detected_angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = theta * 180.0 / np.pi
                if angle > 90:
                    angle -= 180
                detected_angles.append(angle)
            if not detected_angles:
                return 90.0
            angle_bins = np.histogram(detected_angles, bins=36, range=(-90, 90))[0]
            dominant_angle_idx = int(np.argmax(angle_bins))
            dominant_angle = (dominant_angle_idx * 5) - 90
            return abs(float(dominant_angle))
        except Exception:
            return 90.0

    def _sharpness_score(self, img_gray: np.ndarray) -> float:
        """Variance of Laplacian as a sharpness proxy (higher is sharper)."""
        try:
            return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
        except Exception:
            return 0.0
    
    def apply_perspective_correction(self, image: np.ndarray, obb_info: Dict) -> Optional[np.ndarray]:
        """
        Apply perspective correction using linear transformation
        """
        try:
            self.clear_steps()
            self.add_step("original", image, "Original image region")
            
            # Log OBB info for debugging
            logger.info(f"DEBUG: OBB info - center: ({obb_info['center_x']:.1f}, {obb_info['center_y']:.1f}), "
                       f"size: {obb_info['width']:.1f}x{obb_info['height']:.1f}, "
                       f"rotation: {obb_info.get('rotation', 0.0):.3f} rad ({np.degrees(obb_info.get('rotation', 0.0)):.1f}°)")
            
            # Evaluate multiple hypotheses: sign in {s, -s} and +90° offset
            candidates = []
            signs = [self.rotation_sign]
            if self.auto_sign_disambiguation:
                signs.append(-self.rotation_sign)
            offsets = [0.0, float(np.pi/2.0)]
            for s in signs:
                for off in offsets:
                    local_obb = dict(obb_info)
                    local_obb['rotation'] = float(obb_info.get('rotation', 0.0)) + off
                    tm, (w_o, h_o) = self.calculate_perspective_transform_with_sign(local_obb, image.shape[:2], s)
                    corrected = cv2.warpPerspective(image, tm, (w_o, h_o), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY) if len(corrected.shape) == 3 else corrected
                    orient = self._orientation_score_deg(gray)
                    sharp = self._sharpness_score(gray)
                    candidates.append((orient, -sharp, s, off, corrected, tm, (w_o, h_o)))
            candidates.sort(key=lambda x: (x[0], x[1]))
            best_orient, _, used_sign, used_offset, corrected, transform_matrix, (dst_width, dst_height) = candidates[0]
            logger.info(f"DEBUG: Perspective output {dst_width}x{dst_height}, |dominant_angle|≈{best_orient:.1f}°, sign={used_sign:+.0f}, offset_deg={np.degrees(used_offset):.1f}")
            
            self.add_step("perspective_corrected", corrected,
                          f"Perspective corrected (best of signs)",
                          {"transform_matrix": transform_matrix.tolist(),
                           "output_size": (dst_width, dst_height),
                           "chosen_sign": used_sign,
                           "angle_offset_deg": float(np.degrees(used_offset)),
                           "dominant_angle_abs_deg": float(best_orient)})
            return corrected
            
        except Exception as e:
            logger.error(f"Perspective correction failed: {e}")
            return None
    
    def refine_label_boundaries(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract the actual label boundaries within the corrected region
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            self.add_step("grayscale", gray, "Converted to grayscale for boundary detection")
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            self.add_step("blurred", blurred, "Gaussian blur to reduce noise")
            
            # Apply adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            self.add_step("adaptive_threshold", adaptive_thresh, "Adaptive thresholding")
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            self.add_step("canny_edges", edges, "Canny edge detection")
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            self.add_step("morphology_close", closed_edges, "Morphological closing to connect edges")
            
            # Find contours
            contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Create visualization of contours
            contour_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
            self.add_step("contours_detected", contour_vis, f"Detected {len(contours)} contours")
            
            # Filter contours by area and shape
            min_area = (image.shape[0] * image.shape[1]) * 0.1
            valid_contours = []
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Approximate contour
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 4:  # At least quadrilateral
                        valid_contours.append((contour, area, approx))
            
            if not valid_contours:
                return None
            
            # Get the largest valid contour (likely the label boundary)
            largest_contour, _, approx = max(valid_contours, key=lambda x: x[1])
            
            # Create visualization of selected contour
            selected_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(selected_vis, [largest_contour], -1, (0, 255, 0), 3)
            cv2.polylines(selected_vis, [approx], True, (255, 0, 0), 2)
            self.add_step("selected_contour", selected_vis, "Selected largest valid contour")
            
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = self.order_points(box.astype(np.float32))
            
            # Calculate dimensions for the final crop
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            # Ensure minimum dimensions
            if width < 30 or height < 20:
                return None
            
            # Define destination points for final rectification
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Calculate final transformation
            final_transform = cv2.getPerspectiveTransform(box, dst_points)
            
            # Apply final rectification
            final_rectified = cv2.warpPerspective(
                image, final_transform, (width, height),
                flags=cv2.INTER_CUBIC
            )
            
            self.add_step("final_rectified", final_rectified, 
                         "Final rectified label region",
                         {"dimensions": (width, height),
                          "rotation_corrected": True})
            
            return final_rectified
            
        except Exception as e:
            logger.warning(f"Boundary refinement failed: {e}")
            return None

class AdvancedImageProcessor:
    """Enhanced image preprocessing with visualization"""
    
    def __init__(self):
        self.debug_mode = False
        self.perspective_engine = PerspectiveCorrectionEngine()
        # Optional advanced preprocessing toggles
        self.enable_glare_removal: bool = True
        self.enable_shadow_compensation: bool = True
        self.enable_blur_sharpening: bool = True
        self.superres_enabled: bool = False  # requires cv2.dnn_superres or fallback
        # Try to initialize OpenCV super-resolution if available
        self._sr = None
        try:
            # Requires opencv-contrib-python
            from cv2.dnn_superres import DnnSuperResImpl_create  # type: ignore
            self._sr = DnnSuperResImpl_create()
            # No default model loaded; user can load externally. We'll upscale with INTER_CUBIC if not loaded.
        except Exception:
            self._sr = None
        
    def extract_and_rectify_label(self, image: np.ndarray, detection: Dict) -> Tuple[Optional[np.ndarray], List[ProcessingStep]]:
        """
        Extract label region using OBB and apply perspective correction with visualization
        """
        try:
            # Step 1: Apply initial perspective correction
            corrected_region = self.perspective_engine.apply_perspective_correction(image, detection)
            
            if corrected_region is None:
                return None, []
            
            # Step 2: Skip boundary refinement to avoid distortion (keep it simple)
            # Boundary refinement can be aggressive and distort small text
            # refined_region = self.perspective_engine.refine_label_boundaries(corrected_region)
            
            return corrected_region, self.perspective_engine.processing_steps
                
        except Exception as e:
            logger.warning(f"Label rectification failed: {e}")
            return None, []
    
    def remove_glare(self, image: np.ndarray) -> np.ndarray:
        try:
            if not self.enable_glare_removal:
                return image
            img = image.copy()
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            # Specular highlights: high V, low S
            spec_mask = cv2.inRange(v, 230, 255)
            low_sat = cv2.inRange(s, 0, 40)
            mask = cv2.bitwise_and(spec_mask, low_sat)
            # Expand mask to cover halos
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)
            # Inpaint
            if np.count_nonzero(mask) > 0:
                inpainted = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            else:
                inpainted = img
            return inpainted
        except Exception:
            return image
    
    def compensate_shadows(self, image: np.ndarray) -> np.ndarray:
        try:
            if not self.enable_shadow_compensation:
                return image
            img = image.copy()
            # Illumination correction via large-kernel Gaussian division
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=15, sigmaY=15)
            # Avoid divide-by-zero
            illum = (gray.astype(np.float32) + 1.0) / (blur.astype(np.float32) + 1.0)
            illum = cv2.normalize(illum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Light CLAHE after retinex-style normalization
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
            corrected = clahe.apply(illum)
            if len(img.shape) == 3:
                corrected = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
            return corrected
        except Exception:
            return image
    
    def detect_blur_score(self, image_gray: np.ndarray) -> float:
        try:
            return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())
        except Exception:
            return 0.0
    
    def unsharp_mask(self, image_gray: np.ndarray, amount: float = 1.2, radius: int = 1) -> np.ndarray:
        img = image_gray.astype(np.float32)
        blurred = cv2.GaussianBlur(img, (radius * 2 + 1, radius * 2 + 1), 0)
        sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)
    
    def deblur_or_sharpen(self, image: np.ndarray) -> np.ndarray:
        try:
            if not self.enable_blur_sharpening:
                return image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            score = self.detect_blur_score(gray)
            # Threshold tuned for small text; lower scores indicate blur
            if score < 120.0:
                enhanced = self.unsharp_mask(gray, amount=1.4, radius=1)
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(image.shape) == 3 else enhanced
            return image
        except Exception:
            return image
    
    def super_resolve(self, image: np.ndarray, scale: float = 2.0) -> np.ndarray:
        try:
            if not self.superres_enabled:
                return image
            if self._sr is not None:
                # If a model was not set, fall back to resize
                try:
                    return self._sr.upsample(image)
                except Exception:
                    pass
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        except Exception:
            return image
    
    def preprocess_for_ocr(self, image: np.ndarray) -> Tuple[np.ndarray, List[ProcessingStep]]:
        steps: List[ProcessingStep] = []
        def add(name: str, img: np.ndarray, desc: str, params: Dict = None):
            steps.append(ProcessingStep(name, img.copy(), desc, params or {}))
        try:
            img = image
            add("pre_input", img, "Input to advanced preprocessing")
            img = self.remove_glare(img)
            add("glare_removed", img, "Specular highlights removed")
            img = self.compensate_shadows(img)
            add("shadows_compensated", img, "Illumination corrected")
            img = self.deblur_or_sharpen(img)
            add("deblurred_sharpened", img, "Deblur/sharpen if blurred")
            return img, steps
        except Exception as e:
            logger.warning(f"Advanced preprocess_for_ocr failed: {e}")
            return image, steps
    
    def enhance_image_for_ocr(self, image: np.ndarray) -> Tuple[np.ndarray, List[ProcessingStep]]:
        """
        Apply comprehensive image preprocessing for optimal OCR with step visualization
        """
        enhancement_steps = []
        
        def add_enhancement_step(name: str, img: np.ndarray, desc: str, params: Dict = None):
            enhancement_steps.append(ProcessingStep(name, img.copy(), desc, params or {}))
        
        try:
            add_enhancement_step("input", image, "Input image for OCR enhancement")
            
            # Convert to PIL for advanced processing
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            original_size = pil_image.size
            
            # Resize if too small (OCR works better on larger images) - more conservative
            width, height = pil_image.size
            scale_factor = 1.0
            if width < 200 or height < 60:
                scale_factor = max(200 / width, 60 / height, 1.5)  # Reduced from 2.0
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            resized = np.array(pil_image)
            add_enhancement_step("resized", resized, f"Resized by factor {scale_factor:.2f}", 
                                {"original_size": original_size, "scale_factor": scale_factor})
            
            # Convert to grayscale for processing
            gray_pil = pil_image.convert('L')
            gray_np = np.array(gray_pil)
            add_enhancement_step("grayscale_conversion", gray_np, "Converted to grayscale")
            
            # Gentle contrast enhancement only if needed
            img_mean = np.mean(gray_np)
            contrast_factor = 1.0
            if img_mean < 100 or img_mean > 180:  # Only enhance if too dark or too bright
                contrast_factor = 1.2 if img_mean < 100 else 0.9  # Reduced from 1.5
                enhancer = ImageEnhance.Contrast(gray_pil)
                contrast_enhanced = enhancer.enhance(contrast_factor)
                contrast_np = np.array(contrast_enhanced)
                add_enhancement_step("contrast_enhanced", contrast_np, f"Enhanced contrast (factor: {contrast_factor})")
            else:
                contrast_np = gray_np
                add_enhancement_step("contrast_enhanced", contrast_np, "No contrast enhancement needed")
            
            # Very gentle sharpness enhancement
            enhancer = ImageEnhance.Sharpness(Image.fromarray(contrast_np))
            sharpness_enhanced = enhancer.enhance(1.05)  # Reduced from 1.2
            sharpness_np = np.array(sharpness_enhanced)
            add_enhancement_step("sharpness_enhanced", sharpness_np, "Gentle sharpness enhancement (factor: 1.05)")
            
            # Skip aggressive noise reduction for small text
            # Apply gentle CLAHE only if image has poor contrast
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
            clahe_np = clahe.apply(sharpness_np)
            add_enhancement_step("clahe_applied", clahe_np, "Gentle CLAHE histogram equalization")

            # Try adaptive local binarization (Sauvola) alongside grayscale
            try:
                from skimage.filters import threshold_sauvola
                window = max(15, int(min(clahe_np.shape[:2]) * 0.1) | 1)
                sauvola_thresh = threshold_sauvola(clahe_np, window_size=window, k=0.2)
                bin_sauvola = (clahe_np > sauvola_thresh).astype(np.uint8) * 255
                add_enhancement_step("sauvola_binarized", bin_sauvola, f"Sauvola binarization (window={window})")
                final_enhanced = bin_sauvola
            except Exception:
                final_enhanced = clahe_np
            
            # Skip morphological operations and bilateral filtering that can blur text
            
            return final_enhanced, enhancement_steps
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            original_gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return original_gray, enhancement_steps
    
    def apply_rotation_correction(self, image: np.ndarray) -> Tuple[np.ndarray, float, List[ProcessingStep]]:
        """
        Detect and correct text rotation using Hough line transform with visualization
        """
        rotation_steps = []
        
        def add_rotation_step(name: str, img: np.ndarray, desc: str, params: Dict = None):
            rotation_steps.append(ProcessingStep(name, img.copy(), desc, params or {}))
        
        try:
            add_rotation_step("rotation_input", image, "Input for rotation correction")
            
            # Apply edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            add_rotation_step("edge_detection", edges, "Canny edge detection for line detection")
            
            # Apply Hough line transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            
            # Visualize detected lines
            line_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            detected_angles = []
            
            if lines is not None:
                # Draw lines and calculate angles
                for i, line in enumerate(lines[:20]):  # Consider only first 20 lines
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    
                    # Convert to -90 to 90 degree range
                    if angle > 90:
                        angle -= 180
                    detected_angles.append(angle)
                    
                    # Draw line for visualization
                    if i < 10:  # Draw only first 10 lines to avoid clutter
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))
                        cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                add_rotation_step("hough_lines", line_vis, f"Detected {len(lines)} Hough lines",
                                {"total_lines": len(lines), "angles": detected_angles[:10]})
                
                if detected_angles:
                    # Find the most common angle (likely text orientation)
                    angle_bins = np.histogram(detected_angles, bins=36, range=(-90, 90))[0]
                    dominant_angle_idx = np.argmax(angle_bins)
                    dominant_angle = (dominant_angle_idx * 5) - 90
                    
                    add_rotation_step("angle_analysis", line_vis, 
                                    f"Dominant angle: {dominant_angle:.1f}°",
                                    {"dominant_angle": dominant_angle, "angle_distribution": angle_bins.tolist()})
                    
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
                        
                        add_rotation_step("rotation_corrected", corrected, 
                                        f"Rotation corrected by {dominant_angle:.1f}°",
                                        {"rotation_angle": dominant_angle,
                                         "rotation_matrix": rotation_matrix.tolist()})
                        
                        return corrected, dominant_angle, rotation_steps
            
            # No significant rotation needed
            add_rotation_step("no_correction", image, "No rotation correction needed")
            return image, 0.0, rotation_steps
            
        except Exception as e:
            logger.warning(f"Rotation correction failed: {e}")
            return image, 0.0, rotation_steps

class RobustOCREngine:
    """
    Multi-engine OCR system with confidence-based selection
    """
    
    def __init__(self, languages: List[str] = ['en']):
        self.languages = languages
        self.easyocr_reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
        # Engine mode: 'ensemble' | 'trocr' | 'easyocr' | 'tesseract' | 'prefer_trocr'
        self.engine_mode: str = 'ensemble'
        # TrOCR (transformer-based OCR)
        self.trocr_available = True
        self.trocr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # Small model for speed; can be switched to base/large if needed
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed').to(self.trocr_device)
        except Exception as e:
            logger.warning(f"TrOCR not available: {e}")
            self.trocr_available = False
        
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
            # Try both paragraph modes to improve robustness
            ocr_results: List[Dict] = []
            for paragraph_mode in (True, False):
                results = self.easyocr_reader.readtext(image, detail=1, paragraph=paragraph_mode)
                for result in results:
                    # Handle different EasyOCR result formats
                    if len(result) == 3:
                        bbox, text, confidence = result
                    elif len(result) == 2:
                        bbox, text = result
                        confidence = 0.5
                    else:
                        logger.warning(f"Unexpected EasyOCR result format: {result}")
                        continue
                    if confidence > 0.1:
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
    def trocr_extract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using TrOCR (transformer)."""
        if not getattr(self, 'trocr_available', False):
            return []
        try:
            # Expect RGB PIL
            from PIL import Image as PILImage
            if len(image.shape) == 2:
                pil_img = PILImage.fromarray(image).convert('RGB')
            else:
                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pixel_values = self.trocr_processor(images=pil_img, return_tensors="pt").pixel_values.to(self.trocr_device)
            with torch.inference_mode():
                generated_ids = self.trocr_model.generate(pixel_values, max_length=64)
            text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            text = text.strip()
            if not text:
                return []
            # Confidence proxy not provided by model; assign neutral 0.5
            return [{
                "text": text,
                "confidence": 0.5,
                "bbox": [[0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0,image.shape[0]]],
                "engine": "trocr"
            }]
        except Exception as e:
            logger.error(f"TrOCR extraction failed: {e}")
            return []
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def tesseract_extract(self, image: np.ndarray) -> List[Dict]:
        """Extract text using Tesseract OCR"""
        if not self.tesseract_available:
            return []
            
        try:
            # Try multiple PSM modes for robustness
            psm_modes = [6, 7, 11, 13]
            all_results: List[Dict] = []
            for psm in psm_modes:
                config = f'--psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/() '
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
                n_boxes = len(data['text'])
                for i in range(n_boxes):
                    try:
                        confidence = float(data['conf'][i])
                        text = data['text'][i].strip()
                    except Exception:
                        continue
                    if confidence > 30 and len(text) > 0:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        all_results.append({
                            "text": text,
                            "confidence": confidence / 100.0,
                            "bbox": bbox,
                            "engine": "tesseract"
                        })
            return all_results
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def combined_extract(self, image: np.ndarray) -> List[Dict]:
        """
        Run multiple OCR engines and combine results intelligently
        """
        all_results = []
        mode = getattr(self, 'engine_mode', 'ensemble')
        run_easy = mode in ('ensemble', 'easyocr', 'prefer_trocr')
        run_tess = mode in ('ensemble', 'tesseract', 'prefer_trocr')
        run_trocr = mode in ('ensemble', 'trocr', 'prefer_trocr')

        # Run selected engines
        trocr_results: List[Dict] = []
        if run_easy:
            all_results.extend(self.easyocr_extract(image))
        if run_tess and self.tesseract_available:
            all_results.extend(self.tesseract_extract(image))
        if run_trocr:
            trocr_results = self.trocr_extract(image)
            all_results.extend(trocr_results)
        
        if not all_results:
            return []
        
        # Optional simple language/character sanity filter to reduce garbage
        def looks_reasonable(txt: str) -> bool:
            t = txt.strip()
            if len(t) < 2:
                return False
            # Require at least 40% alnum
            alnum = sum(c.isalnum() for c in t)
            if alnum < max(2, int(0.4 * len(t))):
                return False
            return True

        all_results = [r for r in all_results if looks_reasonable(r.get("text", ""))]
        if not all_results:
            return []

        # Merge overlapping results and select best confidence
        merged_results = self.merge_overlapping_results(all_results)
        
        # Filter by confidence and length (raised slightly)
        filtered_results = []
        for result in merged_results:
            text = result["text"].strip()
            if len(text) >= 2 and result["confidence"] > 0.45:
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
        # Ensure angle alignment can be configured
        # Default assumes YOLO's theta is CCW from +x. If your model exports CW, set to -1 via settings endpoint.
        self.image_processor.perspective_engine.set_rotation_sign(1.0)
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
        self.enable_rotation_correction = False
        self.debug_mode = False
        # Multi-scale OCR settings
        self.enable_multiscale_ocr: bool = True
        self.ocr_scales: List[float] = [0.5, 1.0, 1.5, 2.0]
        self.enable_superres: bool = False
        
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
    
    def process_detection_advanced(self, image: np.ndarray, detection: Dict) -> Tuple[List[Dict], Dict[str, Any]]:
        """Process detection with comprehensive visualization of all steps"""
        try:
            preprocess_start = time.time()
            
            all_processing_info = {
                "perspective_steps": [],
                "enhancement_steps": [],
                "rotation_steps": [],
                "timing": {}
            }
            # Ensure rotation_steps variable exists for debug image handling
            rotation_steps: List[ProcessingStep] = []
            
            # Step 1: Extract and rectify label region
            perspective_start = time.time()
            extracted_region, perspective_steps = self.image_processor.extract_and_rectify_label(image, detection)
            perspective_end = time.time()
            
            all_processing_info["perspective_steps"] = [
                {
                    "name": step.name,
                    "description": step.description,
                    "params": step.params,
                    "image_shape": step.image.shape
                } for step in perspective_steps
            ]
            all_processing_info["timing"]["perspective_correction"] = perspective_end - perspective_start
            
            if extracted_region is None or extracted_region.size == 0:
                logger.warning(f"Invalid extracted region for detection: {detection}")
                return [], all_processing_info
            
            # Validate image dimensions
            if extracted_region.shape[0] < 10 or extracted_region.shape[1] < 10:
                logger.warning(f"Extracted region too small: {extracted_region.shape}")
                return [], all_processing_info
            
            # Step 2: Preprocess and multi-scale OCR
            enhancement_start = time.time()
            candidate_images: List[Tuple[np.ndarray, List[ProcessingStep], float]] = []
            base_img = extracted_region
            if self.enable_preprocessing:
                # Advanced pre-processing pipeline first (glare/shadow/blur)
                base_pre, base_adv_steps = self.image_processor.preprocess_for_ocr(base_img)
                # Create scaled variants
                scales = self.ocr_scales if self.enable_multiscale_ocr else [1.0]
                for sc in scales:
                    # Adaptive: skip extreme upscales for already large regions
                    h, w = base_pre.shape[:2]
                    if sc > 1.0 and max(h, w) > 400 and sc >= 2.0:
                        continue
                    scaled = cv2.resize(base_pre, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_CUBIC)
                    if self.enable_superres and sc >= 1.5 and max(h, w) < 140:
                        scaled = self.image_processor.super_resolve(scaled, scale=2.0)
                    # Classic OCR enhancement on the scaled image
                    enhanced, enh_steps = self.image_processor.enhance_image_for_ocr(scaled)
                    # Track steps
                    steps_merged = base_adv_steps + enh_steps
                    candidate_images.append((enhanced, steps_merged, sc))
            else:
                # No preprocessing: pass raw ROI at 1.0 scale directly to OCR
                candidate_images.append((base_img, [], 1.0))
            
            # Collect step images if debug
            if self.debug_mode and candidate_images:
                all_processing_info.setdefault("step_images", {})
                for idx, (img_cand, steps, sc) in enumerate(candidate_images[:3]):
                    for step in steps[:4]:
                        try:
                            _, buffer = cv2.imencode('.jpg', step.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            img_b64 = base64.b64encode(buffer).decode('utf-8')
                            all_processing_info["step_images"][f"enh_{idx}_s{sc:.1f}_{step.name}"] = img_b64
                        except Exception:
                            pass
            enhancement_end = time.time()
            all_processing_info["timing"]["enhancement"] = enhancement_end - enhancement_start
            
            preprocess_end = time.time()
            self.preprocessing_times.append(preprocess_end - preprocess_start)
            if len(self.preprocessing_times) > 100:
                self.preprocessing_times.pop(0)
            
            # Step 4: Perform robust OCR (multi-scale aggregation)
            ocr_start = time.time()
            aggregated: List[Dict] = []
            for enhanced_region, _steps, sc in candidate_images:
                results_at_scale = self.ocr_engine.combined_extract(enhanced_region)
                for r in results_at_scale:
                    r["_scale"] = sc
                aggregated.extend(results_at_scale)
            # Merge across scales
            merged = self.ocr_engine.merge_overlapping_results(aggregated) if aggregated else []
            ocr_results = merged
            ocr_end = time.time()
            
            self.ocr_times.append(ocr_end - ocr_start)
            if len(self.ocr_times) > 100:
                self.ocr_times.pop(0)
            
            all_processing_info["timing"]["ocr"] = ocr_end - ocr_start
            all_processing_info["timing"]["total_preprocessing"] = preprocess_end - preprocess_start
            
            # Step 5: Post-process results
            processed_results = self.post_process_ocr_results(ocr_results)
            
            # Store processing step images for visualization
            if self.debug_mode:
                logger.info(f"DEBUG: Debug mode is True, generating step images")
                all_processing_info["step_images"] = {}
                
                # Store perspective correction steps
                for step in perspective_steps:
                    _, buffer = cv2.imencode('.jpg', step.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    img_b64 = base64.b64encode(buffer).decode('utf-8')
                    all_processing_info["step_images"][f"perspective_{step.name}"] = img_b64
                    logger.info(f"DEBUG: Added step image: perspective_{step.name} ({len(img_b64)} chars)")
                
                # Store enhancement steps (from multi-scale pipeline best-effort)
                # We already inserted a subset earlier for candidates; keep compatibility here by skipping
                
                # Store rotation steps
                if self.enable_rotation_correction:
                    for step in rotation_steps:
                        _, buffer = cv2.imencode('.jpg', step.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        img_b64 = base64.b64encode(buffer).decode('utf-8')
                        all_processing_info["step_images"][f"rotation_{step.name}"] = img_b64
                        logger.info(f"DEBUG: Added step image: rotation_{step.name} ({len(img_b64)} chars)")
                
                logger.info(f"DEBUG: Total step images generated: {len(all_processing_info['step_images'])}")
            else:
                logger.info(f"DEBUG: Debug mode is False, skipping step image generation")
            
            return processed_results, all_processing_info
            
        except Exception as e:
            logger.error(f"Advanced processing failed: {e}")
            return [], {"error": str(e)}
    
    def post_process_ocr_results(self, ocr_results: List[Dict]) -> List[Dict]:
        """Clean up and validate OCR results"""
        if not ocr_results:
            return []
        
        processed = []
        
        # Simple lexicon of expected tokens for product labels; extend as needed
        common_tokens = set([
            "product", "label", "labels", "use", "best", "before", "saturday", "sunday",
            "monday", "tuesday", "wednesday", "thursday", "friday", "ingredients", "barcode",
            "weight", "grams", "kg", "ml", "lot", "batch", "date", "exp", "expiry",
            "manufactured", "by", "for", "store", "fresh", "trash", "blue"
        ])
        
        def token_score(text: str) -> float:
            parts = [p for p in ''.join(c if c.isalnum() or c==' ' else ' ' for c in text.lower()).split() if p]
            if not parts:
                return 0.0
            hits = sum(1 for p in parts if p in common_tokens)
            return hits / len(parts)
        
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
            
            # Light lexicon-based scoring; keep neutral if unknown
            score = token_score(text)
            if score == 0 and len(text) < 4:
                # very short and no known tokens → likely noise
                continue
            
            result["text"] = text
            processed.append(result)
        
        return processed
    
    def create_comprehensive_visualization(self, original_image: np.ndarray, 
                                        detections: List[Dict], 
                                        ocr_results: Dict[int, List[Dict]],
                                        processing_info: Dict[int, Dict[str, Any]]) -> np.ndarray:
        """Create comprehensive visualization showing all processing steps"""
        try:
            # Create a large canvas for comprehensive visualization
            canvas_height = original_image.shape[0] + 600
            canvas_width = max(original_image.shape[1], 1600)
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # Place original annotated image
            annotated = self.create_annotated_image(original_image, detections, ocr_results)
            canvas[:original_image.shape[0], :original_image.shape[1]] = annotated
            
            # Add processing step visualizations
            if self.debug_mode and processing_info:
                y_offset = original_image.shape[0] + 20
                step_height = 120
                step_width = 160
                
                for detection_idx, proc_info in processing_info.items():
                    if detection_idx >= 4:  # Limit to first 4 detections
                        break
                    
                    x_offset = 10 + (detection_idx * (canvas_width // 4))
                    current_y = y_offset
                    
                    # Add detection label
                    cv2.putText(canvas, f"Detection {detection_idx}", (x_offset, current_y - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Display step images if available
                    if "step_images" in proc_info:
                        step_count = 0
                        for step_name, img_b64 in proc_info["step_images"].items():
                            if step_count >= 4:  # Limit steps per detection
                                break
                            
                            try:
                                # Decode image
                                img_data = base64.b64decode(img_b64)
                                nparr = np.frombuffer(img_data, np.uint8)
                                step_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                # Resize for display
                                step_img_resized = cv2.resize(step_img, (step_width, step_height))
                                
                                # Calculate position
                                step_x = x_offset + (step_count % 2) * (step_width + 10)
                                step_y = current_y + (step_count // 2) * (step_height + 30)
                                
                                # Place step image
                                if step_y + step_height < canvas_height and step_x + step_width < canvas_width:
                                    canvas[step_y:step_y+step_height, step_x:step_x+step_width] = step_img_resized
                                    
                                    # Add step label
                                    step_label = step_name.replace('_', ' ').title()
                                    if len(step_label) > 15:
                                        step_label = step_label[:12] + "..."
                                    
                                    cv2.putText(canvas, step_label, (step_x, step_y - 5), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                                
                                step_count += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to display step {step_name}: {e}")
                    
                    # Add timing information
                    timing_info = proc_info.get("timing", {})
                    timing_y = current_y + 280
                    timing_lines = [
                        f"Perspective: {timing_info.get('perspective_correction', 0)*1000:.1f}ms",
                        f"Enhancement: {timing_info.get('enhancement', 0)*1000:.1f}ms", 
                        f"Rotation: {timing_info.get('rotation_correction', 0)*1000:.1f}ms",
                        f"OCR: {timing_info.get('ocr', 0)*1000:.1f}ms"
                    ]
                    
                    for i, line in enumerate(timing_lines):
                        cv2.putText(canvas, line, (x_offset, timing_y + i * 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 0), 1)
            
            return canvas
            
        except Exception as e:
            logger.error(f"Comprehensive visualization failed: {e}")
            return self.create_annotated_image(original_image, detections, ocr_results)
    
    def create_annotated_image(self, image: np.ndarray, detections: List[Dict], 
                             ocr_results: Dict[int, List[Dict]]) -> np.ndarray:
        """Create annotated image with enhanced visualization"""
        annotated = image.copy()
        
        for i, detection in enumerate(detections):
            try:
                # Draw YOLO OBB using the same corner computation as perspective engine
                center_x = float(detection['center_x'])
                center_y = float(detection['center_y'])
                width = float(detection['width'])
                height = float(detection['height'])
                rotation = float(detection['rotation'])
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                # Ensure consistency with preprocessing angle normalization
                points = self.image_processor.perspective_engine.obb_to_corners(
                    center_x, center_y, width, height, rotation
                ).astype(int)
                
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
            "message": "Connected to Enhanced YOLO+OCR Pipeline Server v2.0",
            "device": self.device,
            "yolo_model": self.yolo_model_path,
            "ocr_languages": self.ocr_engine.languages,
            "ocr_enabled": self.enable_ocr,
            "preprocessing_enabled": self.enable_preprocessing,
            "rotation_correction_enabled": self.enable_rotation_correction,
            "tesseract_available": self.ocr_engine.tesseract_available,
            "total_requests_processed": self.total_requests,
            "features": {
                "perspective_correction": True,
                "linear_transformations": True,
                "step_visualization": True,
                "comprehensive_preprocessing": True
            }
        }
        try:
            serializable_welcome = ensure_json_serializable(welcome_msg)
            await websocket.send_text(json.dumps(serializable_welcome, cls=NumpyEncoder))
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def process_image(self, websocket: WebSocket, image_data: str):
        """Process image through enhanced YOLO+OCR pipeline with comprehensive visualization"""
        try:
            total_start_time = time.time()
            
            # Decode image
            try:
                img_data = base64.b64decode(image_data)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_error:
                logger.error(f"Image decode error: {decode_error}")
                error_response = {
                    "type": "error",
                    "message": f"Failed to decode image: {str(decode_error)}"
                }
                serializable_error = ensure_json_serializable(error_response)
                await websocket.send_text(json.dumps(serializable_error, cls=NumpyEncoder))
                return
            
            if img is None:
                error_response = {
                    "type": "error",
                    "message": "Failed to decode image - invalid format"
                }
                serializable_error = ensure_json_serializable(error_response)
                await websocket.send_text(json.dumps(serializable_error, cls=NumpyEncoder))
                return
            
            # Run YOLO inference
            yolo_start = time.time()
            try:
                results = self.yolo_model.predict(img, imgsz=640, verbose=False)
            except Exception as yolo_error:
                logger.error(f"YOLO inference error: {yolo_error}")
                error_response = {
                    "type": "error",
                    "message": f"YOLO inference failed: {str(yolo_error)}"
                }
                serializable_error = ensure_json_serializable(error_response)
                await websocket.send_text(json.dumps(serializable_error, cls=NumpyEncoder))
                return
            yolo_end = time.time()
            
            # Extract YOLO detections
            try:
                detection_data = self.extract_yolo_detections(results)
                detections = detection_data["obb"]
            except Exception as extract_error:
                logger.error(f"YOLO extraction error: {extract_error}")
                error_response = {
                    "type": "error", 
                    "message": f"Detection extraction failed: {str(extract_error)}"
                }
                serializable_error = ensure_json_serializable(error_response)
                await websocket.send_text(json.dumps(serializable_error, cls=NumpyEncoder))
                return
            
            # Perform enhanced OCR processing with comprehensive visualization
            ocr_results = {}
            processing_info = {}
            ocr_start = time.time()
            
            if self.enable_ocr and detections:
                for i, detection in enumerate(detections):
                    try:
                        # Use advanced processing pipeline with visualization
                        ocr_result, proc_info = self.process_detection_advanced(img, detection)
                        
                        if ocr_result:
                            ocr_results[i] = ocr_result
                        
                        processing_info[i] = proc_info
                        
                        # Debug logging
                        if "step_images" in proc_info:
                            logger.info(f"DEBUG: Detection {i} has {len(proc_info['step_images'])} step images")
                        else:
                            logger.info(f"DEBUG: Detection {i} has NO step images")
                                
                    except Exception as ocr_error:
                        logger.warning(f"Enhanced OCR failed for detection {i}: {ocr_error}")
                        processing_info[i] = {"error": str(ocr_error)}
                        
            ocr_end = time.time()
            
            # Create comprehensive visualization
            try:
                if self.debug_mode:
                    annotated_img = self.create_comprehensive_visualization(img, detections, ocr_results, processing_info)
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
            
            # Prepare enhanced response with processing visualization
            response = {
                "type": "pipeline_result",
                "yolo_detections": detections,
                "ocr_results": ocr_results,
                "processing_steps": processing_info,
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
                    "perspective_correction_applied": True,
                    "linear_transformations_used": True,
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
                # Debug: Check if step_images are in the response
                total_step_images = 0
                for i, proc_info in processing_info.items():
                    if isinstance(proc_info, dict) and "step_images" in proc_info:
                        total_step_images += len(proc_info["step_images"])
                        
                logger.info(f"DEBUG: Sending response with {total_step_images} total step images across {len(processing_info)} detections")
                
                # Ensure all values are JSON serializable
                serializable_response = ensure_json_serializable(response)
                
                # Check serializable response size
                response_json = json.dumps(serializable_response, cls=NumpyEncoder)
                logger.info(f"DEBUG: Response JSON size: {len(response_json)} characters")
                
                await websocket.send_text(response_json)
                logger.info(f"Enhanced pipeline processed: {len(detections)} detections, {total_ocr_count} OCR results with comprehensive visualization")
            except Exception as send_error:
                logger.error(f"Failed to send response: {send_error}")
                
        except Exception as e:
            logger.error(f"Enhanced pipeline processing error: {e}")
            try:
                error_response = {
                    "type": "error",
                    "message": f"Enhanced pipeline processing failed: {str(e)}"
                }
                serializable_error = ensure_json_serializable(error_response)
                await websocket.send_text(json.dumps(serializable_error, cls=NumpyEncoder))
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
                                
                                # Normalize rotation atan2 style where needed will be handled in PerspectiveCorrectionEngine
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
app = FastAPI(title="Enhanced YOLO+OCR Pipeline Server v2.0", version="2.0.0")

# Enable CORS for web client settings calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = None
pipeline_settings = {}

def get_pipeline():
    global pipeline, pipeline_settings
    if pipeline is None:
        # Create pipeline with default model path and languages
        pipeline = EnhancedYOLOOCRPipeline()
        # Apply any stored settings
        for key, value in pipeline_settings.items():
            setattr(pipeline, key, value)
    return pipeline

def set_pipeline_setting(key, value):
    """Store a setting to be applied to the pipeline"""
    global pipeline_settings
    pipeline_settings[key] = value
    # Also apply to current pipeline if it exists
    if pipeline is not None:
        setattr(pipeline, key, value)

@app.get("/")
async def get():
    """Serve enhanced server info page"""
    server = get_pipeline()
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced YOLO+OCR Pipeline Server v2.0</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: rgba(255,255,255,0.95); padding: 30px; border-radius: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.2); color: #333; }}
            .status {{ padding: 20px; margin: 20px 0; border-radius: 10px; }}
            .success {{ background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: #2d5016; border-left: 5px solid #28a745; }}
            .info {{ background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #0c3c26; border-left: 5px solid #17a2b8; }}
            .feature {{ background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #fd7e14; }}
            h1 {{ color: #2c3e50; margin-bottom: 25px; text-align: center; font-size: 2.5em; }}
            h3 {{ color: #34495e; margin-top: 30px; font-size: 1.4em; }}
            ul {{ margin-left: 25px; }}
            li {{ margin: 10px 0; line-height: 1.6; }}
            .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin: 25px 0; }}
            .highlight {{ background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            .new-feature {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffd700; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Enhanced YOLO+OCR Pipeline Server v2.0</h1>
            
            <div class="status success">
                ✅ Enhanced pipeline server v2.0 running with advanced linear transformation and comprehensive visualization
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
            
            <div class="new-feature">
                <h3>🆕 Version 2.0 New Features:</h3>
                <ul>
                    <li><strong>Linear Transformation Perspective Correction:</strong> Proper perspective correction using linear transformations</li>
                    <li><strong>Comprehensive Step Visualization:</strong> Visual tracking of every preprocessing step</li>
                    <li><strong>Advanced Boundary Detection:</strong> Refined label boundary detection within corrected regions</li>
                    <li><strong>Step-by-Step Processing Info:</strong> Detailed information about each processing stage</li>
                </ul>
            </div>
            
            <h3>🔧 Enhanced Features:</h3>
            
            <div class="feature">
                <strong>🎯 Advanced Perspective Correction:</strong>
                <ul>
                    <li><strong>Proper Point Ordering:</strong> Top-left, top-right, bottom-right, bottom-left ordering</li>
                    <li><strong>Linear Transformation Matrix:</strong> Accurate perspective transformation using cv2.getPerspectiveTransform</li>
                    <li><strong>OBB-Based Extraction:</strong> Uses oriented bounding box parameters for precise region extraction</li>
                    <li><strong>Boundary Refinement:</strong> Secondary boundary detection within corrected regions</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>📊 Comprehensive Visualization:</strong>
                <ul>
                    <li><strong>Step-by-Step Images:</strong> Visual representation of each processing step</li>
                    <li><strong>Processing Timeline:</strong> Timing information for each processing stage</li>
                    <li><strong>Parameter Tracking:</strong> Storage of transformation matrices and parameters</li>
                    <li><strong>Debug Mode Visualization:</strong> Comprehensive canvas showing all processing steps</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>🔄 Advanced Image Processing:</strong>
                <ul>
                    <li><strong>Gaussian Blur:</strong> Noise reduction before edge detection</li>
                    <li><strong>Adaptive Thresholding:</strong> Dynamic threshold calculation</li>
                    <li><strong>Canny Edge Detection:</strong> Precise edge detection for boundary finding</li>
                    <li><strong>Morphological Operations:</strong> Edge connection and cleanup</li>
                    <li><strong>Contour Analysis:</strong> Shape-based filtering and selection</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>📸 Enhanced Image Preprocessing:</strong>
                <ul>
                    <li><strong>Intelligent Scaling:</strong> Automatic resizing for optimal OCR performance</li>
                    <li><strong>Contrast Enhancement:</strong> PIL-based contrast improvement</li>
                    <li><strong>Sharpness Enhancement:</strong> Edge sharpening for better text clarity</li>
                    <li><strong>Noise Reduction:</strong> Median filtering and bilateral filtering</li>
                    <li><strong>CLAHE:</strong> Contrast Limited Adaptive Histogram Equalization</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>🔄 Rotation Correction with Visualization:</strong>
                <ul>
                    <li><strong>Hough Line Transform:</strong> Line detection for text orientation</li>
                    <li><strong>Angle Analysis:</strong> Statistical analysis of detected line angles</li>
                    <li><strong>Visual Line Display:</strong> Visualization of detected lines</li>
                    <li><strong>Rotation Matrix Application:</strong> Proper geometric transformation</li>
                </ul>
            </div>
            
            <div class="feature">
                <strong>🧠 Multi-Engine OCR with Merging:</strong>
                <ul>
                    <li><strong>EasyOCR Integration:</strong> GPU-accelerated OCR with paragraph detection</li>
                    <li><strong>Tesseract Integration:</strong> Traditional OCR engine with custom configuration</li>
                    <li><strong>Result Merging:</strong> Intelligent combination of results from multiple engines</li>
                    <li><strong>Confidence-Based Selection:</strong> Best result selection based on confidence scores</li>
                </ul>
            </div>
            
            <h3>📈 Processing Pipeline:</h3>
            <div class="highlight">
                <strong>Stage 1:</strong> YOLO OBB Detection → <strong>Stage 2:</strong> Perspective Correction (Linear Transform) → 
                <strong>Stage 3:</strong> Boundary Refinement → <strong>Stage 4:</strong> Image Enhancement → 
                <strong>Stage 5:</strong> Rotation Correction → <strong>Stage 6:</strong> Multi-Engine OCR → 
                <strong>Stage 7:</strong> Result Merging & Visualization
            </div>
            
            <h3>🔍 Debug Mode Features:</h3>
            <ul>
                <li><strong>Step Images:</strong> Base64-encoded images of each processing step</li>
                <li><strong>Transformation Matrices:</strong> Complete transformation matrix data</li>
                <li><strong>Processing Parameters:</strong> All parameters used in each step</li>
                <li><strong>Timing Breakdown:</strong> Detailed timing for each processing stage</li>
                <li><strong>Comprehensive Visualization:</strong> Large canvas showing all steps and results</li>
            </ul>
            
            <h3>📊 Performance Metrics:</h3>
            <ul>
                <li><strong>Perspective Correction Timing:</strong> Time spent on linear transformations</li>
                <li><strong>Enhancement Timing:</strong> Image preprocessing performance</li>
                <li><strong>Rotation Correction Timing:</strong> Text orientation correction time</li>
                <li><strong>OCR Timing:</strong> Multi-engine OCR processing time</li>
                <li><strong>Total Pipeline Timing:</strong> End-to-end processing performance</li>
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
        "rotation_sign": getattr(server.image_processor.perspective_engine, 'rotation_sign', 1.0),
        "auto_sign_disambiguation": getattr(server.image_processor.perspective_engine, 'auto_sign_disambiguation', True),
        "enable_multiscale_ocr": server.enable_multiscale_ocr,
        "ocr_scales": server.ocr_scales,
        "enable_superres": server.enable_superres,
        "ocr_engine_mode": getattr(server.ocr_engine, 'engine_mode', 'ensemble'),
        "tesseract_available": server.ocr_engine.tesseract_available,
        "device": server.device,
        "version": "2.0.0",
        "features": {
            "linear_transformations": True,
            "perspective_correction": True,
            "step_visualization": True,
            "boundary_refinement": True
        }
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
    if "rotation_sign" in settings:
        # Allow runtime control of angle sign convention: 1 or -1
        try:
            sign = float(settings["rotation_sign"])
            server.image_processor.perspective_engine.set_rotation_sign(sign)
        except Exception as _:
            pass
    if "enable_multiscale_ocr" in settings:
        server.enable_multiscale_ocr = bool(settings["enable_multiscale_ocr"])
    if "ocr_scales" in settings:
        try:
            scales = [float(s) for s in settings["ocr_scales"]]
            server.ocr_scales = [s for s in scales if 0.3 <= s <= 3.0]
        except Exception:
            pass
    if "enable_superres" in settings:
        server.enable_superres = bool(settings["enable_superres"])
    if "min_confidence" in settings:
        server.min_confidence = float(settings["min_confidence"])
    if "debug_mode" in settings:
        server.debug_mode = bool(settings["debug_mode"])
    if "ocr_engine_mode" in settings:
        mode = str(settings["ocr_engine_mode"]).lower()
        if mode in ("ensemble", "trocr", "easyocr", "tesseract", "prefer_trocr"):
            server.ocr_engine.engine_mode = mode
        
    return {"status": "updated", "settings": await get_settings()}

@app.options("/settings")
async def options_settings():
    # Handled by CORSMiddleware; explicit route prevents 405 on some clients
    return {"status": "ok"}

@app.get("/debug")
async def get_debug_client():
    """Serve debug client for testing processing steps"""
    with open("debug_client.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/processing-info")
async def get_processing_info():
    """Get detailed information about the processing pipeline"""
    server = get_pipeline()
    return {
        "pipeline_stages": [
            {
                "stage": 1,
                "name": "YOLO Detection",
                "description": "Object detection using oriented bounding boxes",
                "outputs": ["center_x", "center_y", "width", "height", "rotation", "confidence"]
            },
            {
                "stage": 2,
                "name": "Perspective Correction",
                "description": "Linear transformation using cv2.getPerspectiveTransform",
                "methods": ["point_ordering", "rotation_matrix", "perspective_transform"]
            },
            {
                "stage": 3,
                "name": "Boundary Refinement",
                "description": "Advanced edge detection and contour analysis",
                "methods": ["gaussian_blur", "adaptive_threshold", "canny_edges", "morphology", "contour_filtering"]
            },
            {
                "stage": 4,
                "name": "Image Enhancement",
                "description": "Multi-step image preprocessing for OCR optimization",
                "methods": ["scaling", "contrast_enhancement", "sharpness_enhancement", "noise_reduction", "clahe"]
            },
            {
                "stage": 5,
                "name": "Rotation Correction",
                "description": "Text orientation correction using Hough line transform",
                "methods": ["edge_detection", "hough_lines", "angle_analysis", "rotation_transform"]
            },
            {
                "stage": 6,
                "name": "Multi-Engine OCR",
                "description": "Text extraction using multiple OCR engines",
                "engines": ["EasyOCR", "Tesseract"] if server.ocr_engine.tesseract_available else ["EasyOCR"]
            },
            {
                "stage": 7,
                "name": "Result Processing",
                "description": "Confidence-based result merging and visualization",
                "methods": ["overlap_detection", "text_similarity", "confidence_ranking"]
            }
        ],
        "performance_stats": {
            "avg_yolo_time": np.mean(server.inference_times) * 1000 if server.inference_times else 0,
            "avg_ocr_time": np.mean(server.ocr_times) * 1000 if server.ocr_times else 0,
            "avg_preprocessing_time": np.mean(server.preprocessing_times) * 1000 if server.preprocessing_times else 0,
            "total_requests": server.total_requests
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint for pipeline processing"""
    server = get_pipeline()
    await server.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data.strip() == "ping":
                pong_response = {
                    "type": "pong",
                    "timestamp": time.time(),
                    "server_info": {
                        "version": "2.0.0",
                        "preprocessing_enabled": server.enable_preprocessing,
                        "rotation_correction_enabled": server.enable_rotation_correction,
                        "debug_mode": server.debug_mode,
                        "linear_transformations": True,
                        "step_visualization": True
                    }
                }
                serializable_pong = ensure_json_serializable(pong_response)
                await websocket.send_text(json.dumps(serializable_pong, cls=NumpyEncoder))
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
    """Run the enhanced server v2.0"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced YOLO+OCR Pipeline Server v2.0")
    parser.add_argument("--yolo-model", default="runs/obb/train4/weights/best.pt",
                       help="Path to YOLO model file")
    parser.add_argument("--ocr-languages", nargs='+', default=['en'],
                       help="OCR languages (e.g., en zh)")
    parser.add_argument("--host", default="192.168.1.99", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with step visualization")
    parser.add_argument("--no-preprocessing", action="store_true", help="Disable image preprocessing")
    parser.add_argument("--no-rotation-correction", action="store_true", help="Disable rotation correction")
    
    args = parser.parse_args()
    
    global pipeline
    pipeline = EnhancedYOLOOCRPipeline(args.yolo_model, args.ocr_languages)
    
    # Apply command line settings
    pipeline.debug_mode = args.debug
    pipeline.enable_preprocessing = not args.no_preprocessing
    pipeline.enable_rotation_correction = not args.no_rotation_correction
    
    logger.info(f"Starting Enhanced YOLO+OCR Pipeline Server v2.0 on {args.host}:{args.port}")
    logger.info("="*80)
    logger.info("🚀 ENHANCED FEATURES v2.0:")
    logger.info("  ✅ Linear Transformation Perspective Correction")
    logger.info("  ✅ Comprehensive Step-by-Step Visualization")
    logger.info("  ✅ Advanced Boundary Detection and Refinement")
    logger.info("  ✅ Processing Parameter Tracking")
    logger.info("  ✅ Enhanced Debug Mode with Full Pipeline Visualization")
    logger.info("="*80)
    logger.info(f"YOLO Model: {args.yolo_model}")
    logger.info(f"OCR Languages: {args.ocr_languages}")
    logger.info(f"Device: {pipeline.device}")
    logger.info(f"Tesseract Available: {pipeline.ocr_engine.tesseract_available}")
    logger.info(f"Preprocessing Enabled: {pipeline.enable_preprocessing}")
    logger.info(f"Rotation Correction: {pipeline.enable_rotation_correction}")
    logger.info(f"Debug Mode (Step Visualization): {pipeline.debug_mode}")
    logger.info("="*80)
    
    uvicorn.run("enhanced_yolo_ocr_pipeline_v2:app", host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()