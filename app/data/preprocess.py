import os
import cv2

def convert_to_pw(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if not os.path.isfile(input_path):
            continue

        # Load image
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Convert to pure black & white
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Prepare output filename
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_bw{ext}")

        # Save
        cv2.imwrite(output_path, bw)
