As far as I can tell, Label Studio does not presently have a tool to export keypoints with bounding boxes to the YOLO format.

So, convert.py takes the standard format Label Studio .json output file, parses out the relevant bounding box and keypoint data, and exports the data to an output directory of your choice where each file will be a YOLO coordinate file for a single image.

Right now, the convert.py file has the keypoint labels hard-coded, so you will need to change them when you are switching models.

TODO: 
- Auto-evaluate fields with a mapping file (no hard-coding)
