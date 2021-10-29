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
        plt.clf()
        return True
    plt.clf()
    return False

def add_c(name):
    index = name.rfind('.')
    return name[:index] + 'c' + name[index:]

def is_cropped(name):
    print(name[name.rfind('.')-1])
    if(name[name.rfind('.')-1] == 'c'):
        return True
    return False

def find_start_point(df):
    for i in range(0, len(df)):
        if not(is_cropped(df.iloc[i]['Image'])):
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

# do our thing
start_point = find_start_point(dataset)
for i in range(start_point,len(dataset)):
    print(images_processed)
    folder = dataset.iloc[i]["Folder"]
    image = dataset.iloc[i]["Image"]
    source = dataset.iloc[i]["Source"]
    newfolder = "processed_images/"
    # Load the image
    # PIL method
    img = Image.open(folder+image)
    mode = img.mode
    imgformat = img.format
    size = img.size
    del(img)
    # matplotlib method
    img = mpimg.imread(folder+image)
    # Check if its from the same source
    if(source == prevsource)&(source != 'ASM Dataset'):
        if(same_source(img, crop_point)):
            img1 = img[0:round(crop_point[0][1]),:]
            if(imgformat == 'TIFF'):
                image = convert_filetype(image)
            if((mode == 'L') | (mode == 'P') | (imgformat == 'TIFF')):
                newcmap = 'gray'
            else:
                newcmap = None
            mpimg.imsave(newfolder+add_c(image),img1,cmap=newcmap)
            dataset.at[i,'Folder'] = newfolder
            dataset.at[i,'Image'] = add_c(image)
            images_processed = images_processed + 1
            prevsource = source
            if images_processed%5 == 0:
                dataset.to_csv("temp2.csv")
                print("Autosaved")
            continue
    # Display the image
    plt.imshow(img)
    plt.pause(0.01)
    # Prompt user for input
    crop_point = plt.ginput(n=-1, timeout=-1)
    # Close the image
    plt.clf()
    # Display cropped image
    img1 = img[0:round(crop_point[0][1]),:]
    plt.imshow(img1)
    plt.pause(0.01)
    # Save the image
    if((mode == 'L') | (mode == 'P') | (imgformat == 'TIFF')):
        newcmap = 'gray'
    else:
        newcmap = None
    # If we have a tiff, save as png instead
    if(imgformat == 'TIFF'):
        image = convert_filetype(image)
    mpimg.imsave(newfolder+add_c(image),img1, cmap=newcmap)
    images_processed = images_processed + 1
    dataset.at[i,'Folder'] = newfolder
    dataset.at[i,'Image'] = add_c(image)
    print(dataset.at[i,'Folder'])
    prevsource = source
    if images_processed%5 == 0:
        dataset.to_csv("temp2.csv")
        print("Autosaved")

dataset.to_csv("temp2.csv")
print("Finished")
