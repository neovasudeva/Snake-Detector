import os
import time
from PIL import Image, ImageOps

# creates replicas of unlabeled images by flipping them across y-axis 
def flip_images():
    """
    GOAL: take all input images in unlabeled training data and flip them
    """
    # get all files from unlabeled data set
    filenames = [files for path, dirs, files in os.walk("../training/unlabeled/")][0]

    # flip images and save them
    for file in filenames:
        # get name of file without ".jpg"
        index = file.find(".")

        # flip image and save
        im = Image.open("../training/unlabeled/" + file)
        flipped = ImageOps.mirror(im)
        flipped.save("../training/unlabeled-flipped/" + file[ : index] + "-f.jpg")

flip_images()
