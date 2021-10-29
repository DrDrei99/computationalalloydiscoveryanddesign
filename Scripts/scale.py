import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import sys

def get_scale(image):
    scale = 0
    img = mpimg.imread(image)
    plt.imshow(img)
    plt.pause(0.001)
    scale = input("Enter the scale of the image (um): ")
    plt.clf()
    return scale


dataset = pd.read_csv("temp.csv")
image_count = 0

for i in range(0,len(dataset)):
    if np.isnan(dataset.iloc[i]['Scale']):
        folder = dataset.iloc[i]["Folder"]
        image = dataset.iloc[i]["Image"]
        scale = get_scale(folder + image)
        image_count = image_count+1
        print(scale)
        if scale == 'save':
            dataset.to_csv("temp2.csv")
            sys.exit()
        print(image)
        print("No. of images processed: "+ str(image_count))
        dataset.at[i,'Scale'] = scale
        if image_count%10 == 0:
            dataset.to_csv("temp2.csv")
            print("Autosaved")
