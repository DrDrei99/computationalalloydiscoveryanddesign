import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



def rotate90deg(img):
    angle = -90
    rotatedimg = img.rotate(angle, expand=True)
    return rotatedimg

def get4quads(img):
    imgarray = np.asarray(img)
    
    dims = img.size
    midpoints = [0,0]
    midpoints[0] = round(dims[0]/2)
    midpoints[1] = round(dims[1]/2)

    squaresizes = min([midpoints[0],dims[0]-midpoints[0],midpoints[1],dims[1]-midpoints[1]])
    
    quad1 = imgarray[midpoints[1]-squaresizes:midpoints[1],midpoints[0]-squaresizes:midpoints[0]]
    quad4 = imgarray[midpoints[1]:midpoints[1]+squaresizes,midpoints[0]:midpoints[0]+squaresizes]
    quad2 = imgarray[midpoints[1]-squaresizes:midpoints[1],midpoints[0]:midpoints[0]+squaresizes]
    quad3 = imgarray[midpoints[1]:midpoints[1]+squaresizes,midpoints[0]-squaresizes:midpoints[0]]

    quad1img = Image.fromarray(quad1)
    quad2img = Image.fromarray(quad2)
    quad3img = Image.fromarray(quad3)
    quad4img = Image.fromarray(quad4)
    
    return quad1img, quad2img, quad3img, quad4img

def to224x224(img):
    curr_size = img.size
    if (curr_size[0] != curr_size[1]):
        print("This aint a square")
        print(curr_size)
    resized_image = img.resize((224,224))
    scale_factor = 224/curr_size[0]
    return resized_image, scale_factor

#quad1img.save(new_folder+imagename+"q1.png")
#quad2img.save(new_folder+imagename+"q2.png")
#quad3img.save(new_folder+imagename+"q3.png")
#quad4img.save(new_folder+imagename+"q4.png")
#plt.imshow(img1)
#plt.show()

'''
What we got: All sorts of image sizes, some RGB, some grayscale
What we need: 224x224x3 images in grayscale to eliminate bias

How we go about it:
1. Load image
2. Check the size of the image, split into 4 quadrants
find biggest possible squares
3. Resize each quadrant to 224x224, rescale scale bar
4. Make sure each image has 3 channels in grayscale
5. Add something to identify each quadrant
6. Save image name with path to dataset
'''

df = pd.read_csv("cropped_dataset.csv")

#make a new dataframe to store our processed images
newdf = pd.DataFrame(data=None, index = pd.RangeIndex(start=0, stop = len(df)*16, step = 1), columns=df.columns)

# main loop
imagetracker = 0

for i in range(0,len(df)):

    newdfi = i*16
    folder = df['Folder'][i]
    image = df['Image'][i]
    imagename = image[0:image.rfind('.')]

    source = df['Source'][i]
    name = df['Name'][i]
    strength = df['Tensile Strength (Yield)'][i]
    scale_length= df['Scale Length (px)'][i]
    scale = df['Scale'][i]
    estimate = df['Estimate'][i]
    note = df['Note'][i]
    prop_source= df['Prop Source'][i]

    new_folder = 'final_images/'
    #new_name = 'test_image'
    img = Image.open(folder+image)

    quads = get4quads(img)

    for quad in range(0,4):
        quadindex = newdfi + (quad * 4)
        currentimage, scale_factor = to224x224(quads[quad])
        for rotation in range(0,4):

            imagetracker = imagetracker + 1
            print("At image no." + str(imagetracker))

            if (rotation != 0):
                currentimage = rotate90deg(currentimage)
            rotationindex = quadindex + rotation
            newimagename = imagename + 'Q' + str(quad+1) + 'R' + str(rotation+1) + '.png'
            newdf.at[rotationindex,'Source']= source
            newdf.at[rotationindex,'Folder'] = new_folder
            newdf.at[rotationindex,'Image'] = newimagename
            newdf.at[rotationindex,'Name']= name
            newdf.at[rotationindex,'Tensile Strength (Yield)']= strength
            newdf.at[rotationindex,'Scale Length (px)']= scale_length * scale_factor
            newdf.at[rotationindex,'Scale']= scale
            newdf.at[rotationindex,'Estimate']= estimate
            newdf.at[rotationindex,'Note']= note
            newdf.at[rotationindex,'Prop Source']= prop_source    
            currentimage.save(new_folder+newimagename)        

#imagename = image[0:image.rfind('.')]

#q1.save(new_folder+imagename+'Q1R1.png')
#q2.save(new_folder+imagename+'Q2R1.png')
#q3.save(new_folder+imagename+'Q3R1.png')
#q4.save(new_folder+imagename+'Q4R1.png')
#
#q1 = rotate90deg(q1)
#q1.save(new_folder+imagename+'Q1R2.png')
#
#q1 = rotate90deg(q1)
#q1.save(new_folder+imagename+'Q1R3.png')
#
#q1 = rotate90deg(q1)
#q1.save(new_folder+imagename+'Q1R4.png')

newdf.to_csv('final_dataset.csv')