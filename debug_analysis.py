import streamlit as st
import tempfile
import os
import traceback
from pathlib import Path
from PIL import Image
import numpy as np

st.title("Debug Analysis")

uploaded_file = st.file_uploader("Upload test image", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    st.write("File uploaded successfully")
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    st.write(f"Temporary file created: {temp_path}")
    
    try:
        # Test 1: Basic file operations
        st.write("Testing basic file operations...")
        file_size = os.path.getsize(temp_path)
        st.write(f"File size: {file_size} bytes")
        
        # Test 2: PIL image loading
        st.write("Testing PIL image loading...")
        image = Image.open(temp_path)
        st.write(f"Image size: {image.size}")
        st.write(f"Image mode: {image.mode}")
        
        # Test 3: Convert to RGB and numpy
        st.write("Converting to RGB and numpy array...")
        if image.mode != 'RGB':
            image = image.convert('RGB')
        pixels = np.array(image)
        st.write(f"Pixels shape: {pixels.shape}")
        
        # Test 4: Try entropy calculation
        st.write("Testing entropy calculation...")
        from utils.file_analysis import calculate_entropy
        entropy = calculate_entropy(temp_path)
        st.write(f"Entropy: {entropy}")
        
        # Test 5: Try metadata extraction
        st.write("Testing metadata extraction...")
        from utils.file_analysis import get_file_metadata
        metadata = get_file_metadata(temp_path)
        st.write(f"Metadata keys: {list(metadata.keys())[:5]}")  # Show first 5 keys
        
        # Test 6: Try steganography detection
        st.write("Testing steganography detection...")
        from utils.stego_detector import analyze_image_for_steganography
        with st.spinner("Running steganography analysis..."):
            detection_result = analyze_image_for_steganography(temp_path)
            st.write(f"Detection likelihood: {detection_result.likelihood:.2f}")
        
        st.success("All tests completed successfully!")
        
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.code(traceback.format_exc())
    
    finally:
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass