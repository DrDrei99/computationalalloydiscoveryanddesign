import numpy as np
from keras.models import sequential, load_model
from keras.layers import Dense
from keras.layers import InputLayer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

featurevectors = pd.read_csv("vggoutputs.csv")
dataset = pd.read_csv("final_dataset.csv")

note = ['Defaced', 'Bar'] 
estimate = ['Condition', 'Unspecified', 'cold rolled', 'hot rolled']

dataset = dataset.loc[~(dataset['Estimate'].isin(estimate) | dataset['Note'].isin(note))]

imagenames = dataset['Image'].values
baseimages = list(set([x[0:x.rfind("Q")+2] for x in imagenames]))

train, test = train_test_split(baseimages, test_size=0.7)

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

model = load_model("ML/Tuner/optimal_model_untrained")

epoch_num = 60
batch_num = 15

model.fit(X_train, y_train, epochs=epoch_num, batch_size = batch_num)

#_, accuracy = model.evaluate(X_train, y_train)
#print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict(X_train)
train_predictions = [round(x[0]) for x in predictions]

plt.plot(y_train, train_predictions, 'ro', markersize=2, label = 'Seen/Training Images')

predictions = model.predict(X_test)
test_predictions = [round(x[0]) for x in predictions]

plt.plot(y_test, test_predictions, 'bo', markersize=2, label = 'Validation/Unseen Images')

plt.legend()
plt.title("Performance of evaluating test data")
plt.xlabel("Expected Tensile Yield Strength (MPa)")
plt.ylabel("Predicted Tensile Yield Strength (MPa)")
plt.show()

predictions = test_predictions.append(train_predictions)
split = ['Test' for x in testimages].append(['Train' for x in trainimages])

if (input("Save these outputs? (y/n)\n")) == 'y':
    filename = 'optimalhyper'
    results = pd.DataFrame()
    results['Image'] = np.append(testimages, trainimages)
    results['Expected'] = np.append(y_test, y_train)
    results['Predicted'] = predictions
    results['Subset'] = split
    results.to_csv("ML/Results/"+filename+'.csv', index=False)

pass