import os 
import cv2 
import numpy as np 

from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

class ImageCaptionOverlay:
    @staticmethod
    def add_caption_overlay(image: np.ndarray, caption:str, position:str="bottom",
    font_size:int=1, thickness: int=2, font_path:str|None=None)-> np.ndarray:
        img_copy = image.copy()
        height, width = img_copy.shape[:2]

        # If custom font is provided, use PIL for rendering
        if font_path and os.path.exists(font_path):
            # Convert to PIL for custom font rendering
            pil_image = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            try:
                # Scale font_size appropriately (convert from CV2 scale to pixel size)
                pil_font_size = int(font_size * 20)
                pil_font = ImageFont.truetype(font_path, pil_font_size)
            except Exception:
                pil_font = ImageFont.load_default()

            # Calculate text dimensions and wrap if needed
            max_width = width - 40
            bbox = draw.textbbox((0, 0), caption, font=pil_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            if text_width > max_width:
                words = caption.split()
                lines = []
                current_line = ""

                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    test_bbox = draw.textbbox((0, 0), test_line, font=pil_font)
                    test_width = test_bbox[2] - test_bbox[0]

                    if test_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word

                if current_line:
                    lines.append(current_line)
            else:
                lines = [caption]

            # Calculate positions
            line_height = text_height + 10
            total_height = len(lines) * line_height

            if position.lower() == "bottom":
                start_y = height - total_height - 20
            elif position.lower() == "top":
                start_y = 30
            else:  # Center
                start_y = (height - total_height) // 2

            # Draw text with background
            for i, line in enumerate(lines):
                bbox = draw.textbbox((0, 0), line, font=pil_font)
                line_width = bbox[2] - bbox[0]
                line_height_actual = bbox[3] - bbox[1]
                text_x = (width - line_width) // 2
                text_y = start_y + (i * line_height)

                # Draw background rectangle
                draw.rectangle(
                    [text_x - 10, text_y - 5, text_x + line_width + 10, text_y + line_height_actual + 5],
                    fill=(0, 0, 0)
                )

                # Draw text
                draw.text((text_x, text_y), line, fill=(66, 140, 255), font=pil_font)

            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        else:
            # Use OpenCV's built-in font (original implementation)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Calculate text size and position
            text_size = cv2.getTextSize(caption, font, font_size, thickness)[0]

            # Wrap the text if too long
            max_width = width - 40
            if text_size[0] > max_width:
                words = caption.split()
                lines = []
                current_line = ""

                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    test_size = cv2.getTextSize(test_line, font, font_size, thickness)[0]

                    if test_size[0] <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word

                if current_line:
                    lines.append(current_line)
            else:
                lines = [caption]

            # Calculate Positions
            line_height = cv2.getTextSize("A", font, font_size, thickness)[0][1] + 10
            total_height = len(lines) * line_height

            if position.lower() == "bottom":
                start_y = height - total_height - 20
            elif position.lower() == "top":
                start_y = 30
            else:  # Center
                start_y = (height - total_height) // 2

            # Add background rectangle for better readability
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_size, thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = start_y + (i * line_height) + text_size[1]

                # Background Rectangle
                cv2.rectangle(
                    img_copy, (text_x - 10, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 10, text_y + 5),
                    (0, 0, 0), -1
                )

                # Text
                cv2.putText(img_copy, line, (text_x, text_y), font, font_size, (255, 140, 66), thickness)

            return img_copy
    
    @staticmethod
    def add_caption_background(image:np.ndarray, caption:str, font_path:str|None=None, font_size:int=24,
    background_color: Tuple=(33, 34, 69), text_color: Tuple=(183, 212, 225), margin:int=50)->np.ndarray:
        height, width = image.shape[:2]

        # Use PIL for better text rendering 
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Try to use custom font or default
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            elif os.path.exists("fonts/Poppins-Regular.ttf"):
                font = ImageFont.truetype("fonts/Poppins-Regular.ttf", font_size)
            else:
                # Fallback to default font
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        # Calculate text dimensions 
        draw = ImageDraw.Draw(pil_image)
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Wrap text if necessary 
        max_width = width - (2 * margin)
        if text_width > max_width:
            words = caption.split()
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + " " + word if current_line else word 
                test_bbox = draw.textbbox((0, 0), test_line, font=font)
                test_width = test_bbox[2] - test_bbox[0]

                if test_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)
        else:
            lines = [caption]

        # Calculate total text height 
        total_text_height = len(lines) * text_height + (len(lines) - 1) * 10

        # Create new image with space for text 
        new_height = height + total_text_height + (2 * margin)
        new_image = Image.new("RGB", (width, new_height), background_color)

        # Paste original image 
        new_image.paste(pil_image, (0, int(total_text_height + (2 * margin))))

        # Add Text
        draw = ImageDraw.Draw(new_image)
        y_offset = margin

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x_position = (width - line_width)//2

            draw.text((x_position, y_offset), line, fill=text_color, font=font)
            y_offset += text_height + 10

        # Convert back to OpenCV format 
        return cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)