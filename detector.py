import os
import numpy as np
import json
import cv2

# function to pull JSON data from annotations
def get_snake_data():
    """
    Returns a dictionary of the form { filename, x, y, x2, y2, label }
    """
    # get path to annotations JSON file
    json_file = "./training/labeled/_annotations.json"
    with open(json_file) as f:
        full_annots = json.load(f)

    # create dictionary specifically for annotations
    annotations = full_annots["annotations"]

    # return annotations
    return annotations

# function for testing whether get_snake_data function works
def test_json(annotations):
    """
    Tests get_snake_data() by iterating over bounding boxes in test
    images
    
    Params:
        annotations = dictionary of images and bounding box locations
    """
    for filename in annotations:
        # open image
        img = cv2.imread("./training/labeled/" + str(filename))

        # get dimensions
        height, width, channels = img.shape

        # draw bounding boxes on img
        for box in annotations[filename]:
            # get coordinates
            x1 = int(box["x"] * width)
            x2 = int(box["x2"] * width)
            y1 = int(box["y"] * height)
            y2 = int(box["y2"] * height)

            # draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # show image
        cv2.imshow("snek", img)
        cv2.waitKey()
        cv2.destroyAllWindows()


# visualize dataset
annotations = get_snake_data()
test_json(annotations)



