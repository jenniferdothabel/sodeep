import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import subprocess
import tempfile
import os

st.set_page_config(
    page_title="Test Image Creator",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Test Image Creator")
st.write("Create clean test images and images with hidden steganographic data")

tab1, tab2 = st.tabs(["🖼️ Create Clean Image", "🔐 Embed Hidden Data"])

with tab1:
    st.subheader("Generate Clean Test Image")
    
    col1, col2 = st.columns(2)
    
    with col1:
        width = st.slider("Width", 100, 2000, 800)
        height = st.slider("Height", 100, 2000, 600)
        
        image_type = st.selectbox("Image Type", [
            "Solid Color", 
            "Gradient", 
            "Random Noise",
            "Simple Pattern",
            "Text Image"
        ])
        
        if image_type == "Solid Color":
            color = st.color_picker("Choose Color", "#0066CC")
            
        elif image_type == "Gradient":
            start_color = st.color_picker("Start Color", "#FF0000")
            end_color = st.color_picker("End Color", "#0000FF")
            
        elif image_type == "Text Image":
            text = st.text_input("Text to display", "DEEP ANAL TEST")
            text_color = st.color_picker("Text Color", "#FFFFFF")
            bg_color = st.color_picker("Background Color", "#000000")
    
    with col2:
        if st.button("Generate Clean Image"):
            # Create image based on type
            if image_type == "Solid Color":
                # Convert hex to RGB
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                img = Image.new('RGB', (width, height), rgb)
                
            elif image_type == "Gradient":
                img = Image.new('RGB', (width, height))
                pixels = img.load()
                
                start_rgb = tuple(int(start_color[i:i+2], 16) for i in (1, 3, 5))
                end_rgb = tuple(int(end_color[i:i+2], 16) for i in (1, 3, 5))
                
                for x in range(width):
                    ratio = x / width
                    r = int(start_rgb[0] * (1-ratio) + end_rgb[0] * ratio)
                    g = int(start_rgb[1] * (1-ratio) + end_rgb[1] * ratio)
                    b = int(start_rgb[2] * (1-ratio) + end_rgb[2] * ratio)
                    
                    for y in range(height):
                        pixels[x, y] = (r, g, b)
                        
            elif image_type == "Random Noise":
                # Create random noise
                noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                img = Image.fromarray(noise)
                
            elif image_type == "Simple Pattern":
                img = Image.new('RGB', (width, height), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                
                # Create checkerboard pattern
                square_size = 50
                for x in range(0, width, square_size):
                    for y in range(0, height, square_size):
                        if (x // square_size + y // square_size) % 2:
                            draw.rectangle([x, y, x+square_size, y+square_size], fill=(0, 0, 0))
                            
            elif image_type == "Text Image":
                bg_rgb = tuple(int(bg_color[i:i+2], 16) for i in (1, 3, 5))
                text_rgb = tuple(int(text_color[i:i+2], 16) for i in (1, 3, 5))
                
                img = Image.new('RGB', (width, height), bg_rgb)
                draw = ImageDraw.Draw(img)
                
                # Try to use a default font, fallback to basic if not available
                try:
                    font_size = min(width, height) // 10
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Get text size and center it
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (width - text_width) // 2
                y = (height - text_height) // 2
                
                draw.text((x, y), text, fill=text_rgb, font=font)
            
            # Display the image
            st.image(img, caption="Generated Clean Image")
            
            # Provide download
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            buf.seek(0)
            
            st.download_button(
                label="Download Clean Image",
                data=buf.getvalue(),
                file_name="clean_test_image.png",
                mime="image/png"
            )

with tab2:
    st.subheader("Create Image with Hidden Data")
    
    uploaded_clean = st.file_uploader(
        "Upload a clean image to embed data in",
        type=['png', 'jpg', 'jpeg'],
        help="Upload the clean image you want to hide data in"
    )
    
    if uploaded_clean:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_clean, caption="Original Image")
            
            # Options for hiding data
            hide_method = st.selectbox("Steganography Method", [
                "LSB Text Embedding",
                "Steghide (JPEG only)",
                "Simple LSB"
            ])
            
            if hide_method == "LSB Text Embedding":
                secret_text = st.text_area("Secret message to hide", "This is a secret message from dad")
                
            elif hide_method == "Steghide (JPEG only)":
                secret_text = st.text_area("Secret message to hide", "Secret Navy message")
                password = st.text_input("Password (optional)", type="password")
        
        with col2:
            if st.button("Embed Hidden Data"):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_input:
                    tmp_input.write(uploaded_clean.getvalue())
                    input_path = tmp_input.name
                
                try:
                    if hide_method == "LSB Text Embedding":
                        # Simple LSB embedding
                        img = Image.open(input_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        
                        pixels = list(img.getdata())
                        
                        # Convert message to binary
                        binary_message = ''.join(format(ord(char), '08b') for char in secret_text)
                        binary_message += '1111111111111110'  # End marker
                        
                        # Embed in LSBs
                        data_index = 0
                        new_pixels = []
                        
                        for pixel in pixels:
                            r, g, b = pixel
                            
                            if data_index < len(binary_message):
                                # Modify LSB of red channel
                                r = (r & 0xFE) | int(binary_message[data_index])
                                data_index += 1
                                
                            new_pixels.append((r, g, b))
                        
                        # Create new image
                        stego_img = Image.new('RGB', img.size)
                        stego_img.putdata(new_pixels)
                        
                        # Save to buffer
                        buf = io.BytesIO()
                        stego_img.save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.success("✅ Message embedded successfully!")
                        st.image(stego_img, caption="Image with Hidden Data")
                        
                        st.download_button(
                            label="Download Steganographic Image",
                            data=buf.getvalue(),
                            file_name="stego_test_image.png",
                            mime="image/png"
                        )
                        
                    elif hide_method == "Steghide (JPEG only)":
                        if uploaded_clean.type == "image/jpeg":
                            # Create temp files
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp_msg:
                                tmp_msg.write(secret_text)
                                msg_path = tmp_msg.name
                            
                            output_path = tempfile.mktemp(suffix='.jpg')
                            
                            # Use steghide to embed
                            cmd = ['steghide', 'embed', '-cf', input_path, '-ef', msg_path, '-sf', output_path]
                            if password:
                                cmd.extend(['-p', password])
                            else:
                                cmd.extend(['-p', ''])
                            
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                # Read the output file
                                with open(output_path, 'rb') as f:
                                    stego_data = f.read()
                                
                                st.success("✅ Message embedded with Steghide!")
                                
                                st.download_button(
                                    label="Download Steganographic Image",
                                    data=stego_data,
                                    file_name="stego_steghide_image.jpg",
                                    mime="image/jpeg"
                                )
                                
                                # Cleanup
                                os.unlink(output_path)
                                os.unlink(msg_path)
                            else:
                                st.error(f"Steghide failed: {result.stderr}")
                        else:
                            st.error("Steghide only works with JPEG images")
                            
                except Exception as e:
                    st.error(f"Error embedding data: {str(e)}")
                finally:
                    # Cleanup
                    os.unlink(input_path)

st.markdown("---")
st.write("""
**Instructions:**
1. **Create Clean Image**: Generate a test image without any hidden data
2. **Embed Hidden Data**: Upload a clean image and embed secret messages using steganography
3. **Test Both**: Use these images to test DEEP ANAL's detection accuracy

This helps verify that:
- Clean images show low detection rates (10-30%)
- Images with hidden data show high detection rates (70%+)
""")