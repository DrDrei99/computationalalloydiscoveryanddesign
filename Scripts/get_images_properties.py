import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import sys
from PIL import Image

# This program is meant to collect the image
# modes and formats

# Open our dataset
dataset = pd.read_csv("cropped_dataset.csv")
# make a dict
properties = {}

# Get the image path
for i in range(0,len(dataset)):
    folder = dataset.iloc[i]["Folder"]
    image = dataset.iloc[i]["Image"]

    # Load the image
    # PIL method
    img = Image.open(folder+image)

    # Record what we want
    properties[image] = [img.format, img.mode, img.size[0], img.size[1]]

# Convert our dict to a df
newdf = pd.DataFrame.from_dict(properties,orient='index',
                                columns=['Format','Mode','ResX', 'ResY'])
# Save df
newdf.to_csv('properties.csv')
