import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import sys
from PIL import Image

# This program is meant to display an image from a csv file
# prompt the user for input to decide where to crop it
# and save it back in a different location

# If it's from the same source, ask if we want to use the same
# crop point
def same_source(img, crop_point):
    dispimg = img[0:round(crop_point[0][1]),:]
    plt.imshow(dispimg)
    plt.pause(0.01)
    print("Dimensions are:" + str(dispimg.shape))
    comm = input("Is this where we crop?:")
    if(comm==''):
        return True
    return False

def add_c(name):
    index = name.rfind('.')
    return name[:index] + 'C' + name[index:]

def is_cropped(name):
    if(name[name.rfind('.')-1] == 'C'):
        return True
    return False

def find_start_point(df):
    for i in range(0, len(df)):
        if is_cropped(df.iloc[i]['Image']):
            return i
    return 0
def convert_filetype(name):
    index = name.rfind('.')
    return name[:index] + '.png'

# counters
images_processed = 0

# Open our dataset
dataset = pd.read_csv("temp2.csv")
prevsource = "First"

i=0

folder = dataset.iloc[i]["Folder"]
image = dataset.iloc[i]["Image"]
source = dataset.iloc[i]["Source"]
newfolder = "test_images/"
print(image)
# Load the image
# PIL method
img = Image.open(folder+image)
img1 = mpimg.imread(folder+image)
print(img.size)
plt.imshow(img)
plt.pause(0.01)
crop_point = plt.ginput(n=-1,timeout=-1)

#img.save(newfolder+convert_filetype(image), format='png')
#img = Image.open(newfolder+"test.png")
#
pass