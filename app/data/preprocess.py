import os
import cv2
import typer

app = typer.Typer()

@app.command()
def convert_to_pw(input_dir: str, output_dir: str):
    """
    Convert all images in input_dir to black & white and save to output_dir with _bw suffix.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if not os.path.isfile(input_path):
            continue

        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_bw{ext}")
        cv2.imwrite(output_path, bw)

if __name__ == "__main__":
    app()
