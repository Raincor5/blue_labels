#!/usr/bin/env python3
"""
Test script to check if visualization steps are being generated correctly
"""

import json
import numpy as np
from enhanced_yolo_ocr_pipeline import EnhancedYOLOOCRPipeline

def test_visualization_generation():
    """Test if visualization steps are being generated"""
    print("Testing visualization step generation...")
    
    # Create pipeline with debug mode enabled
    pipeline = EnhancedYOLOOCRPipeline()
    pipeline.debug_mode = True
    pipeline.enable_preprocessing = True
    pipeline.enable_rotation_correction = True
    
    print(f"Debug mode: {pipeline.debug_mode}")
    print(f"Preprocessing enabled: {pipeline.enable_preprocessing}")
    print(f"Rotation correction enabled: {pipeline.enable_rotation_correction}")
    
    # Create a test image
    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    # Create a mock detection
    mock_detection = {
        "center_x": 200.0,
        "center_y": 150.0,
        "width": 100.0,
        "height": 80.0,
        "confidence": 0.85,
        "class_id": 0,
        "class_name": "label"
    }
    
    print("\nProcessing test detection...")
    
    try:
        # Process the detection
        ocr_results, processing_info = pipeline.process_detection_advanced(test_image, mock_detection)
        
        print(f"OCR results: {len(ocr_results)}")
        print(f"Processing info keys: {list(processing_info.keys())}")
        
        # Check if step images are generated
        if "step_images" in processing_info:
            step_images = processing_info["step_images"]
            print(f"Step images generated: {len(step_images)}")
            for step_name, img_b64 in step_images.items():
                print(f"  - {step_name}: {len(img_b64)} chars")
        else:
            print("No step_images found in processing_info")
            
        # Check perspective steps
        if "perspective_steps" in processing_info:
            print(f"Perspective steps: {len(processing_info['perspective_steps'])}")
            
        # Check enhancement steps
        if "enhancement_steps" in processing_info:
            print(f"Enhancement steps: {len(processing_info['enhancement_steps'])}")
            
        # Check rotation steps
        if "rotation_steps" in processing_info:
            print(f"Rotation steps: {len(processing_info['rotation_steps'])}")
            
        # Test JSON serialization
        try:
            serializable_info = pipeline.ensure_json_serializable(processing_info)
            json_str = json.dumps(serializable_info, cls=pipeline.NumpyEncoder)
            print(f"JSON serialization successful: {len(json_str)} chars")
        except Exception as e:
            print(f"JSON serialization failed: {e}")
            
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization_generation() 