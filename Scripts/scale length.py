import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import sys

def get_scale_length(image):
    length = 0
    img = mpimg.imread(image)
    plt.imshow(img) #aspect='auto'
    plt.pause(0.001)
    positions = plt.ginput(n=-1, timeout=-1)
    length = calculate_distance(positions)
    plt.clf()
    return length

def calculate_distance(positions):
    if abs(positions[0][1]-positions[1][1])<100:
        length = abs(positions[0][0]-positions[1][0])
        return round(length)
    length = abs(positions[0][1]-positions[1][1])
    return round(length)

dataset = pd.read_csv("temp.csv")

#folder = dataset.iloc[0]["Folder"]
#image = dataset.iloc[0]["Image"]
#img = mpimg.imread(folder+image)
#plt.imshow(img,aspect='auto')
#positions = plt.ginput(n=-1, timeout=-1)
#print(str(positions))

image_count = 0

for i in range(0,len(dataset)):
    if np.isnan(dataset.iloc[i]['Scale Length (px)']):
        folder = dataset.iloc[i]["Folder"]
        image = dataset.iloc[i]["Image"]
        scale_length = get_scale_length(folder + image)
        image_count = image_count+1
        print(scale_length)
        print(image)
        print("No. of images processed: "+ str(image_count))
        dataset.at[i,'Scale Length (px)'] = scale_length
        if image_count%5 == 0:
            dataset.to_csv("temp2.csv")
            print("Autosaved")
        #command = input("Save?")
        #if command == 'y':
        #    dataset.to_csv("temp2.csv")
        #    sys.exit()
        
dataset.to_csv("temp2.csv")
print("Finished")