#!/usr/bin/env python3
"""
Debug script to check visualization data being sent from the server.
"""
import json
import base64
import cv2
import numpy as np
from enhanced_yolo_ocr_pipeline import EnhancedYOLOOCRPipeline, NumpyEncoder, ensure_json_serializable

def test_processing_steps_generation():
    """Test if processing steps and step images are being generated correctly"""
    print("Testing processing steps generation...")
    
    # Create pipeline with debug mode enabled
    pipeline = EnhancedYOLOOCRPipeline()
    pipeline.debug_mode = True
    
    # Create a test image
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 128
    cv2.rectangle(test_image, (50, 50), (350, 250), (255, 255, 255), -1)
    cv2.putText(test_image, "TEST", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Create a mock detection with all required fields
    mock_detection = {
        "center_x": 200.0,
        "center_y": 150.0,
        "width": 300.0,
        "height": 200.0,
        "rotation": 0.0,  # Add the missing rotation field
        "confidence": 0.95,
        "class_id": 0,
        "class_name": "test_label"
    }
    
    try:
        # Process the detection
        ocr_results, processing_info = pipeline.process_detection_advanced(test_image, mock_detection)
        
        print(f"OCR Results: {len(ocr_results)}")
        print(f"Processing Info Keys: {list(processing_info.keys())}")
        
        # Check if step_images are present
        if "step_images" in processing_info:
            step_images = processing_info["step_images"]
            print(f"Step Images Keys: {list(step_images.keys())}")
            print(f"Number of step images: {len(step_images)}")
            
            # Check a few step images
            for key, img_b64 in list(step_images.items())[:3]:
                print(f"  {key}: {len(img_b64)} chars (base64)")
                
                # Try to decode and verify it's a valid image
                try:
                    img_data = base64.b64decode(img_b64)
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    print(f"    Decoded image shape: {decoded_img.shape}")
                except Exception as e:
                    print(f"    Error decoding image: {e}")
        else:
            print("❌ No step_images found in processing_info")
            
        # Test JSON serialization
        test_response = {
            "type": "pipeline_result",
            "processing_steps": {0: processing_info},
            "yolo_detections": [mock_detection],
            "ocr_results": {0: ocr_results}
        }
        
        try:
            serialized = json.dumps(ensure_json_serializable(test_response), cls=NumpyEncoder)
            print(f"✅ JSON serialization successful: {len(serialized)} chars")
            
            # Test deserialization
            deserialized = json.loads(serialized)
            if "processing_steps" in deserialized:
                proc_steps = deserialized["processing_steps"]
                if "0" in proc_steps and "step_images" in proc_steps["0"]:
                    print(f"✅ Deserialized step_images: {len(proc_steps['0']['step_images'])} images")
                else:
                    print("❌ No step_images in deserialized data")
            else:
                print("❌ No processing_steps in deserialized data")
                
        except Exception as e:
            print(f"❌ JSON serialization failed: {e}")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

def test_web_client_data_structure():
    """Test the data structure that the web client expects"""
    print("\nTesting web client data structure...")
    
    # Create mock data that matches what the server sends
    mock_step_images = {
        "perspective_original": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",  # 1x1 pixel
        "enhancement_input": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        "rotation_rotated": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    }
    
    mock_processing_steps = {
        "0": {
            "step_images": mock_step_images,
            "perspective_steps": [
                {"name": "original", "description": "Original image", "params": {}},
                {"name": "transformed", "description": "Perspective corrected", "params": {}}
            ],
            "enhancement_steps": [
                {"name": "input", "description": "Input for enhancement", "params": {}},
                {"name": "enhanced", "description": "Enhanced image", "params": {}}
            ],
            "rotation_steps": [
                {"name": "rotated", "description": "Rotation corrected", "params": {}}
            ]
        }
    }
    
    # Test the web client's createStepHTML logic
    def simulate_createStepHTML(step, stepId, stepImages):
        """Simulate the web client's createStepHTML function"""
        imageSrc = None
        if stepImages:
            possibleKeys = [
                step["name"],
                f"perspective_{step['name']}",
                f"enhancement_{step['name']}",
                f"rotation_{step['name']}"
            ]
            
            for key in possibleKeys:
                if key in stepImages:
                    imageSrc = f"data:image/jpeg;base64,{stepImages[key]}"
                    break
        
        return {
            "step_name": step["name"],
            "image_found": imageSrc is not None,
            "image_src": imageSrc
        }
    
    # Test each step
    for step_type, steps in [
        ("perspective", mock_processing_steps["0"]["perspective_steps"]),
        ("enhancement", mock_processing_steps["0"]["enhancement_steps"]),
        ("rotation", mock_processing_steps["0"]["rotation_steps"])
    ]:
        print(f"\nTesting {step_type} steps:")
        for step in steps:
            result = simulate_createStepHTML(step, "test_id", mock_step_images)
            print(f"  {step['name']}: image_found={result['image_found']}")

if __name__ == "__main__":
    print("🔍 Debugging visualization issues...\n")
    test_processing_steps_generation()
    test_web_client_data_structure()
    print("\n✅ Debug tests completed!") 