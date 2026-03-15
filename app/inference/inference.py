import cv2 # Computer vision package import - has drawing tools so we can print an image to the screen
import math # Only used for square root in absolute_distance formula because I'm lazy

# from IPython import embed # Ignore this, used for debugging
from pathlib import Path # Used for grabbing local directory names on filesystem
from ultralytics import YOLO # The machine learning package import

""" Load the YOLO weights for the SCALE model and the TADPOLE model """
scale_spliner = YOLO("app/models/scale_model_ouptut/run_20260314_170124/pose/train/weights/best.pt")
tadpole_spliner = YOLO("app/models/spline_model_ouptut/run_20260314_170124/pose/train/weights/best.pt")

""" Absolute distance formula sqrt((x2-x1)^2 + (y2-y1)^2 )"""
def absolute_distance(tuple1, tuple2):
    return math.sqrt(((tuple1[0]-tuple2[0])**2) + ((tuple1[1] - tuple2[1]) ** 2))

""" Process a single image file through the machine learned models """
def process(file:str):

    # SCALE model inferencing
    scale_result = scale_spliner(file)
    s = scale_result[0]

    # TADPOLE model inferencing
    tadpole_result = tadpole_spliner(file)
    r = tadpole_result[0]

    # Store a temporary reference to the image of the TADPOLE model's plot (raw image file plus bounding box and dots)
    img = r.plot()

    # Break out the TADPOLE keypoint coordinates (basically I'm just ignoring the other metadata in the YOLO object)
    tadpole_kp_coords = r.keypoints.xy[0].cpu().numpy()  # convert to numpy floats

    # The keypoints don't come with our labeled names, so here I just create a dictionary for NAME : DATA
    labeled_tadpole_kp_coords = {
        "pos_rostrum": tadpole_kp_coords[0],
        "pos_tailtip": tadpole_kp_coords[1],
        "pos_tailbase": tadpole_kp_coords[2],
        "pos_tailbase_third": tadpole_kp_coords[3],
        "pos_tailtip_third": tadpole_kp_coords[4],
    }

    ## Calculate the deltas between points using our predefined absolute distance formula
    # I call them "del" because we don't really care what they're called since we're just going to sum them. Later we may care.
    labeled_tadpole_kp_deltas = {
        "del1": absolute_distance(labeled_tadpole_kp_coords["pos_rostrum"], labeled_tadpole_kp_coords["pos_tailbase"]),
        "del2": absolute_distance(labeled_tadpole_kp_coords["pos_tailbase"], labeled_tadpole_kp_coords["pos_tailbase_third"]),
        "del3": absolute_distance(labeled_tadpole_kp_coords["pos_tailbase_third"], labeled_tadpole_kp_coords["pos_tailtip_third"]),
        "del4": absolute_distance(labeled_tadpole_kp_coords["pos_tailtip_third"], labeled_tadpole_kp_coords["pos_tailtip"]),
    }

    # Here I just create circles with radius 24px, color red (0,0,255) and position them in the buffered img.
    for x, y in tadpole_kp_coords:
        cv2.circle(img, (int(x), int(y)), 24, (0,0,255), -1)

    # Tuple array to store the information about the lines between the vertices. This is just more art stuff
    tadpole_connections = [(0,2), (2,3), (3,4), (4,1)]

    for a, b in tadpole_connections:
        x1, y1 = tadpole_kp_coords[a]
        x2, y2 = tadpole_kp_coords[b]

        cv2.line(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0,255,0),
            12
        ) # Line width 12, green line (0,255,0)

    # Same thing as the TADPOLE model return data: drop the metadata
    ruler_kp = s.keypoints.xy[0].cpu().numpy()  # convert to numpy

    # Unlike the named dictionary before, we only care about the x-y coordinates of each of these ruler markers, so store them in a simple array
    ruler_deltas = [
        absolute_distance(ruler_kp[0], ruler_kp[1]),
        absolute_distance(ruler_kp[1], ruler_kp[2]),
        absolute_distance(ruler_kp[2], ruler_kp[3]),
        absolute_distance(ruler_kp[3], ruler_kp[4])
    ]

    # Simple mean
    mean_ruler_del = sum(ruler_deltas) / len(ruler_deltas)

    # Divide the sum of all tadpole segment lengths by the mean ruler delta to give us units of ruler ticks (presumed 1mm)
    tadpole_length_mm = sum(labeled_tadpole_kp_deltas.values()) / mean_ruler_del

    # Draw similar dots on the ruler markings
    for x, y in ruler_kp:
        cv2.circle(img, (int(x), int(y)), 24, (0,0,255), -1)

    # Draw connecting lines between ruler dots
    for i in range(len(ruler_kp)-1):
        x1, y1 = ruler_kp[i]
        x2, y2 = ruler_kp[i+1]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 12)

    # Labeling the image with the tadpole length rounded to two "very roughly approximated sig-figs"
    text = f"Tadpole Length {round(tadpole_length_mm, 2)} mm"
    org = (50, 250)           # x, y position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)  # white
    thickness = 8

    # Draw text
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    # Finally, actually render the image
    cv2.imshow("keypoints_bigger", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Our main function called on launch - our "starting place"
def main():

    print("Hello from tadpolemetry!")

    # Loop through all files in production/input directory
    for file_path in Path("production/input").iterdir():
        try:
            if file_path.is_file():

                # execute the process function for one file at a time
                process(file_path)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
