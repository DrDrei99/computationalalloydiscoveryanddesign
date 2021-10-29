import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#def convertPNGtoGrayscale(img):
#    shape = img.size
#    imgarray = np.asarray(img)
#    for i in range(0,shape[0])
#        for z in range(0,shape[1])
#        r = imgarray[i][z][0]
#        g = imgarray[i][z][1]
#        b = imgarray[i][z][2]
#        imgarray[i][z][0] = round(np.average(r+g+b))
#        imgarray[i][z][1] = round(np.average(r+g+b))
#        imgarray[i][z][2] = round(np.average(r+g+b))
#    img = Image.fromarray(imgarray)
#    return img

df = pd.read_csv('cropped_dataset.csv')
processed_images = 0
for i in range(0,len(df)):
    print(processed_images)
    folder = df['Folder'][i]
    image = df['Image'][i]

    img = Image.open(folder+image)

    img = img.convert(mode='RGB')

    shape = img.size
    #print(shape)
    imgarray = np.copy(np.asarray(img))
    
    r = np.uint32(imgarray[:,:,0])
    g = np.uint32(imgarray[:,:,1])
    b = np.uint32(imgarray[:,:,2])
    grey = np.uint8((r+g+b)/3)
    imgarray[:,:,0] = grey
    imgarray[:,:,1] = grey
    imgarray[:,:,2] = grey

    img = Image.fromarray(imgarray)

    img.save('converted_images/'+image)
    processed_images = processed_images + 1