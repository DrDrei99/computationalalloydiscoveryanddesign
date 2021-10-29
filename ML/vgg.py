import pandas as pd
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

dataset = pd.read_csv('final_dataset.csv')

model = VGG16(
    include_top = False,
    weights = 'imagenet',
    input_shape = (224, 224, 3),
    pooling = 'max'
)

features_dict = {}

imageno = 0
for i in range(0, len(dataset)):
    print(imageno)
    img_path = dataset['Folder'][i] + dataset['Image'][i]

    img = image.load_img(img_path)
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    features_dict[dataset['Image'][i]] = features[0]

    imageno = imageno + 1

vgg_features = pd.DataFrame.from_dict(features_dict)
vgg_features.to_csv('vggoutputs.csv', index=False)