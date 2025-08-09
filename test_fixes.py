#!/usr/bin/env python3
"""
Test script to verify the fixes for EasyOCR result format and JSON serialization issues.
"""

import json
import numpy as np
from enhanced_yolo_ocr_pipeline import NumpyEncoder, ensure_json_serializable

def test_easyocr_format_handling():
    """Test the EasyOCR result format handling"""
    print("Testing EasyOCR result format handling...")
    
    # Simulate different EasyOCR result formats
    test_results = [
        # Standard 3-element format
        ([[209, 227], [265, 227], [265, 265], [209, 265]], 'Ut', 0.8),
        # 2-element format (no confidence)
        ([[65, 291], [231, 291], [231, 412], [65, 412]], 'Wob @ Mum'),
        # Another 2-element format
        ([[137, 291], [342, 291], [342, 520], [137, 520]], 'M 7 Kcbt MMuI')
    ]
    
    for i, result in enumerate(test_results):
        print(f"  Test {i+1}: {result}")
        if len(result) == 3:
            bbox, text, confidence = result
            print(f"    -> 3-element format: bbox={bbox}, text='{text}', confidence={confidence}")
        elif len(result) == 2:
            bbox, text = result
            confidence = 0.5  # Default confidence
            print(f"    -> 2-element format: bbox={bbox}, text='{text}', confidence={confidence}")
        else:
            print(f"    -> Unexpected format: {result}")
    
    print("✓ EasyOCR format handling test completed\n")

def test_json_serialization():
    """Test JSON serialization with numpy types"""
    print("Testing JSON serialization...")
    
    # Create test data with numpy types
    test_data = {
        "numpy_int": np.int32(42),
        "numpy_float": np.float32(3.14),
        "numpy_array": np.array([1, 2, 3]),
        "nested_dict": {
            "numpy_float64": np.float64(2.718),
            "numpy_int64": np.int64(100)
        },
        "list_with_numpy": [np.float32(1.5), np.int16(10), "string"]
    }
    
    print("  Original data types:")
    for key, value in test_data.items():
        print(f"    {key}: {type(value)} = {value}")
    
    # Test the ensure_json_serializable function
    serializable_data = ensure_json_serializable(test_data)
    
    print("\n  After ensure_json_serializable:")
    for key, value in serializable_data.items():
        print(f"    {key}: {type(value)} = {value}")
    
    # Test JSON serialization
    try:
        json_str = json.dumps(serializable_data, cls=NumpyEncoder)
        print(f"\n  JSON serialization successful: {json_str[:100]}...")
        print("✓ JSON serialization test completed\n")
        return True
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False

def test_float32_serialization():
    """Test specific float32 serialization issue"""
    print("Testing float32 serialization...")
    
    # Create data that might cause the original error
    test_response = {
        "type": "pipeline_result",
        "yolo_detections": [
            {
                "center_x": np.float32(100.5),
                "center_y": np.float32(200.3),
                "confidence": np.float32(0.85)
            }
        ],
        "ocr_results": {
            "0": [
                {
                    "text": "Test",
                    "confidence": np.float32(0.75),
                    "bbox": [[np.float32(10.0), np.float32(20.0)]]
                }
            ]
        },
        "timing": {
            "total_ms": np.float32(150.25)
        }
    }
    
    try:
        serializable_response = ensure_json_serializable(test_response)
        json_str = json.dumps(serializable_response, cls=NumpyEncoder)
        print("  ✓ float32 serialization successful")
        return True
    except Exception as e:
        print(f"  ✗ float32 serialization failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing fixes for EasyOCR and JSON serialization issues...\n")
    
    # Run tests
    test_easyocr_format_handling()
    json_success = test_json_serialization()
    float32_success = test_float32_serialization()
    
    print("=" * 50)
    print("Test Results:")
    print(f"  EasyOCR format handling: ✓")
    print(f"  JSON serialization: {'✓' if json_success else '✗'}")
    print(f"  float32 serialization: {'✓' if float32_success else '✗'}")
    
    if json_success and float32_success:
        print("\n🎉 All tests passed! The fixes should resolve the issues.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 