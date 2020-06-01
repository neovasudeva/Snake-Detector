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
        flipped.save("../training/unlabeled/" + file[ : index] + "-f.jpg")

# resizes and rotates images to the same dimensions
def resize_images():
    """
    GOAL: take all images in unlabeled training data and rotate and resize them 
    to be the same shape and size (510x385)
    """
    # get all files from unlabeled data set
    filenames = [files for path, dirs, files in os.walk("../training/unlabeled/")][0]

    # rotate image if needed, resize them, then save them
    for file in filenames:
        # open image and get dimensions
        image = Image.open("../training/unlabeled/" + file)
        width, height = image.size

        # rotate image if the height is larger than width
        if height > width:
            image = image.transpose(Image.ROTATE_90)

        # resize image to 510x385
        image = image.resize((510, 385))
        
        # save image
        image.save("../training/unlabeled/" + file)

flip_images()
resize_images()



