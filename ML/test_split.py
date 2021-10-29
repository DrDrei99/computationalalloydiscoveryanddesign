import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 


featurevectors = pd.read_csv("vggoutputs.csv")
dataset = pd.read_csv("final_dataset.csv")

imagenames = dataset['Image'].values
baseimages = list(set([x[0:x.rfind("Q")+2] for x in imagenames]))

train, test = train_test_split(baseimages, test_size=0.2)

trainimages = [image for image in imagenames if (image[0:image.rfind("Q")+2] in train)]
testimages = [image for image in imagenames if (image[0:image.rfind("Q")+2] in test)]

featureinputs_train = featurevectors[trainimages].values
featureinputs_test = featurevectors[testimages].values

dataset_train = dataset.loc[dataset["Image"].isin(trainimages)]
dataset_test = dataset.loc[dataset["Image"].isin(testimages)]

scalebar_train = dataset_train['Scale Length (px)'].values
scale_train = dataset_train['Scale'].values
scale_per_pixel_train = scale_train/scalebar_train

scalebar_test = dataset_test['Scale Length (px)'].values
scale_test = dataset_test['Scale'].values
scale_per_pixel_test = scale_test/scalebar_test

expected_strength_train = dataset_train['Tensile Strength (Yield)'].values
expected_strength_test = dataset_test['Tensile Strength (Yield)'].values

featureinputs_train = np.append(featureinputs_train, [scale_per_pixel_train], axis = 0)
featureinputs_train = np.transpose(featureinputs_train)

featureinputs_test = np.append(featureinputs_test, [scale_per_pixel_test], axis = 0)
featureinputs_test = np.transpose(featureinputs_test)

X_train = featureinputs_train
y_train = expected_strength_train

X_test = featureinputs_test
y_test = expected_strength_test


print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)