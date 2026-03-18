import os
import json
import typer
import logging

from IPython import embed

from typing import Annotated

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = typer.Typer()

@app.command()
def main(filename: str, output_dir: str):

    with open(filename) as f:
        data = json.load(f)

    for d in data:
        output_file = output_dir+"/"+os.path.splitext(d['file_upload'])[0]+".txt"

        # Must match 5 for bounding box +  (kpt_shape[a] * kpt_shape[b]) in dataset_n.yaml file
        num_fields = 5 + (5 * 3)
        data_simple = [0 for _ in range(num_fields)] # Simple empty of [0, 0, ... , 0] of size num_fields

        for p in d['annotations'][0]['result']:

            # We expect there to only be one rectangle label per image:
            if p['type'] == 'rectanglelabels':

                # Label Studio tracks pixels in percentages of, e.g., 35.3, 65.1, etc.
                # YOLO tracks in percentages of 0.353, 0.651, etc. so we do simple scaling

                width = p['value']['width'] / 100
                height = p['value']['height'] / 100

                """ Note the fixed index below of each data_simple[n]. These represet the fixed index in the final data file. """

                data_simple[0] = 0 # The YOLO file format expects the class name's index here. In our case, with one class, it's always zero

                # Label Studio tracks bounding box positions by center-of-box, so we need to offset the coordinates by the width and height
                # to get the top-left-of-box coordinates for YOLO
                data_simple[1] = (p['value']['x']/100) + (width/2)
                data_simple[2] = (p['value']['y']/100) + (height/2)

                # Width and height are the same (after converting scale 100x above)
                data_simple[3] = width
                data_simple[4] = height


            else:
                try:
                    label = p['value']['keypointlabels'][0]
                    # TODO: Make this all automatic with a mapping file
                    # TODO: Clean this up with some list comprehension


                    """ these labels are all associated with the scale_model """
                    if label == 'tick_1':
                        data_simple[5] = p['value']['x']/100
                        data_simple[6] = p['value']['y']/100
                        data_simple[7] = 2
                    elif label == 'tick_2':
                        data_simple[8] = p['value']['x']/100
                        data_simple[9] = p['value']['y']/100
                        data_simple[10] = 2
                    elif label == 'tick_3':
                        data_simple[11] = p['value']['x']/100
                        data_simple[12] = p['value']['y']/100
                        data_simple[13] = 2
                    elif label == 'tick_4':
                        data_simple[14] = p['value']['x']/100
                        data_simple[15] = p['value']['y']/100
                        data_simple[16] = 2
                    elif label == 'tick_5':
                        data_simple[17] = p['value']['x']/100
                        data_simple[18] = p['value']['y']/100
                        data_simple[19] = 2
                    elif label == 'rostrum':
                        """ these labels are all associated with the spline_model """
                        data_simple[5] = p['value']['x']/100
                        data_simple[6] = p['value']['y']/100
                        data_simple[7] = 2
                    elif label == 'tailtip':
                        data_simple[8] = p['value']['x']/100
                        data_simple[9] = p['value']['y']/100
                        data_simple[10] = 2
                    elif label == 'tailbase':
                        data_simple[11] = p['value']['x']/100
                        data_simple[12] = p['value']['y']/100
                        data_simple[13] = 2
                    elif label == 'tailbase_third':
                        data_simple[14] = p['value']['x']/100
                        data_simple[15] = p['value']['y']/100
                        data_simple[16] = 2
                    elif label == 'tailtip_third':
                        data_simple[17] = p['value']['x']/100
                        data_simple[18] = p['value']['y']/100
                        data_simple[19] = 2
                    else:
                        logging.error(f"Unknown keypoint label '{label}'")


                except Exception as e:
                    logger.info(f"{e} {label}")
                    input("We've reached an error. We probably shouldn't be here, but just check your output data. Press any key to continue.")

        with open(output_file, "w") as out:
            [out.write(f'{n} ') for n in data_simple]

if __name__ == "__main__":
    app()
