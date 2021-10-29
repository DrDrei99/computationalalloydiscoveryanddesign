import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get4quads(img):
    imgarray = np.asarray(img)
    
    dims = img.size
    midpoints = [round(dims[0]/2),round(dims[1]/2)]

    squaresizes = min([midpoints[0],dims[0]-midpoints[0],midpoints[1],dims[1]-midpoints[1]])
    
    quad1 = imgarray[midpoints[1]-squaresizes:midpoints[1],midpoints[0]-squaresizes:midpoints[0]]
    quad4 = imgarray[midpoints[1]:midpoints[1]+squaresizes,midpoints[0]:midpoints[0]+squaresizes]
    quad2 = imgarray[midpoints[1]-squaresizes:midpoints[1],midpoints[0]:midpoints[0]+squaresizes]
    quad3 = imgarray[midpoints[1]:midpoints[1]+squaresizes,midpoints[0]-squaresizes:midpoints[0]]

    print(quad1.shape)
    print(quad2.shape)    
    print(quad3.shape)
    print(quad4.shape)

    quad1img = Image.fromarray(quad1)
    quad2img = Image.fromarray(quad2)
    quad3img = Image.fromarray(quad3)
    quad4img = Image.fromarray(quad4)
    
    return quad1img, quad2img, quad3img, quad4img

df = pd.read_csv('cropped_dataset.csv')

folder = df['Folder'][0]
image = '355-0-U-5c.png'

img = Image.open(folder+image)

quads = get4quads(img)

for quad in quads:
    print(quad.size)