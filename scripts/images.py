import os
import time
from PIL import Image, ImageOps

# creates replicas of unlabeled images by flipping them across y-axis 
def flip_images(path):
    """
    GOAL: take all input images in unlabeled training data and flip them
    Params:
        path = string path to images
    """
    # get all files from unlabeled data set
    filenames = [files for path, dirs, files in os.walk(path)][0]

    # flip images and save them
    for file in filenames:
        # get name of file without ".jpg"
        index = file.find(".")

        # flip image and save
        im = Image.open(path + file)
        flipped = ImageOps.mirror(im)
        flipped.save(path + file[ : index] + "-f.jpg")

# resizes and rotates images to the same dimensions
def resize_images(path):
    """
    GOAL: take all images in unlabeled training data and rotate and resize them 
    to be the same shape and size (510x385)
    Params:
        path = string path to images
    """
    # get all files from unlabeled data set
    filenames = [files for path, dirs, files in os.walk(path)][0]

    # rotate image if needed, resize them, then save them
    for file in filenames:
        # open image and get dimensions
        image = Image.open(path + file)
        width, height = image.size

        # rotate image if the height is larger than width
        if height > width:
            image = image.transpose(Image.ROTATE_90)

        # resize image to 510x385
        image = image.resize((510, 385))
        
        # save image
        image.save(path + file)

#flip_images("../test/unlabeled/")
resize_images("../test/unlabeled/")



