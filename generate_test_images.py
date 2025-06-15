#!/usr/bin/env python3
"""
Generate test images for steganography detection validation
"""
import numpy as np
from PIL import Image
import argparse
import os

def create_clean_image(width=800, height=600, image_type="solid", output="clean_test.png"):
    """Create a clean test image without any hidden data"""
    
    if image_type == "solid":
        # Solid blue color
        img = Image.new('RGB', (width, height), (0, 102, 204))
        
    elif image_type == "gradient":
        # Simple gradient
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for x in range(width):
            ratio = x / width
            r = int(255 * ratio)
            g = int(128 * (1-ratio))
            b = int(255 * (1-ratio))
            
            for y in range(height):
                pixels[x, y] = (r, g, b)
                
    elif image_type == "noise":
        # Random noise (should trigger high detection if algorithm is too sensitive)
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        
    elif image_type == "checkerboard":
        # Simple pattern
        img = Image.new('RGB', (width, height), (255, 255, 255))
        pixels = img.load()
        
        square_size = 50
        for x in range(width):
            for y in range(height):
                if (x // square_size + y // square_size) % 2:
                    pixels[x, y] = (0, 0, 0)
    
    img.save(output)
    print(f"Clean image saved: {output}")
    return output

def embed_lsb_message(input_image, message, output="stego_test.png"):
    """Embed a message using simple LSB steganography"""
    
    img = Image.open(input_image)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += '1111111111111110'  # End marker
    
    pixels = list(img.getdata())
    
    if len(binary_message) > len(pixels):
        raise ValueError("Message too long for image")
    
    # Embed in LSBs of red channel
    new_pixels = []
    data_index = 0
    
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
    stego_img.save(output)
    
    print(f"Steganographic image saved: {output}")
    print(f"Message embedded: '{message}'")
    return output

def main():
    parser = argparse.ArgumentParser(description='Generate test images for steganography detection')
    parser.add_argument('--clean', action='store_true', help='Generate clean test images')
    parser.add_argument('--stego', action='store_true', help='Generate steganographic test images')
    parser.add_argument('--type', choices=['solid', 'gradient', 'noise', 'checkerboard'], 
                       default='solid', help='Type of clean image to generate')
    parser.add_argument('--message', default='This is a secret test message', 
                       help='Message to embed in steganographic image')
    parser.add_argument('--width', type=int, default=800, help='Image width')
    parser.add_argument('--height', type=int, default=600, help='Image height')
    
    args = parser.parse_args()
    
    if args.clean:
        print("Generating clean test images...")
        
        # Generate different types of clean images
        create_clean_image(args.width, args.height, "solid", "clean_solid.png")
        create_clean_image(args.width, args.height, "gradient", "clean_gradient.png")
        create_clean_image(args.width, args.height, "checkerboard", "clean_checkerboard.png")
        
        # Generate noise image (should test algorithm sensitivity)
        create_clean_image(args.width, args.height, "noise", "clean_noise.png")
        
        print("\nTest these images in DEEP ANAL:")
        print("- clean_solid.png should show LOW detection (10-30%)")
        print("- clean_gradient.png should show LOW detection (10-30%)")
        print("- clean_checkerboard.png should show LOW detection (10-30%)")
        print("- clean_noise.png may show higher detection due to randomness")
    
    if args.stego:
        print("Generating steganographic test images...")
        
        # First create a base image if it doesn't exist
        if not os.path.exists("clean_solid.png"):
            create_clean_image(args.width, args.height, "solid", "clean_solid.png")
        
        # Embed different messages
        embed_lsb_message("clean_solid.png", args.message, "stego_message.png")
        embed_lsb_message("clean_solid.png", "SECRET NAVY DATA", "stego_navy.png")
        embed_lsb_message("clean_solid.png", "A" * 100, "stego_long.png")  # Longer message
        
        print("\nTest these images in DEEP ANAL:")
        print("- stego_message.png should show HIGH detection (70%+)")
        print("- stego_navy.png should show HIGH detection (70%+)")
        print("- stego_long.png should show HIGH detection (70%+)")
    
    if not args.clean and not args.stego:
        print("Use --clean to generate clean images or --stego to generate steganographic images")
        print("Example: python generate_test_images.py --clean --stego")

if __name__ == "__main__":
    main()