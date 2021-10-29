import numpy as np
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from kerastuner.tuners import Hyperband
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

def build_model(hp):
    model = Sequential()
    
    model.add(InputLayer(input_shape=(513,)))
    
    for i in range(hp.Int("num_layers",min_value = 1, max_value = 4, step = 1)):
        model.add(Dense(
            units=hp.Int("units_"+str(i), min_value=32, max_value = 608, step = 32),
            activation='relu'
        ))
    
    model.add(Dense(1, activation='relu'))
    model.compile(loss="mean_squared_error",optimizer="adam", metrics=["mean_squared_error"])
    
    return model


featurevectors = pd.read_csv("vggoutputs.csv")
dataset = pd.read_csv("final_dataset.csv")

note = ['Defaced', 'Bar'] 
estimate = ['Condition', 'Unspecified', 'cold rolled', 'hot rolled']

dataset = dataset.loc[~(dataset['Estimate'].isin(estimate) | dataset['Note'].isin(note))]

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

tuner = Hyperband(
    build_model,
    objective = "val_mean_squared_error",
    max_epochs = 60,
    seed=69,
    directory = 'ML/Tuner',
    project_name = 'Research_project_2',
    overwrite=True
)

tuner.search(X_train, y_train, epochs=60, validation_data=(X_test, y_test))

model = tuner.get_best_models(num_models=1)[0]

model.save("ML/Tuner/second_optimal")

_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict(X_train)
rounded = [round(x[0]) for x in predictions]

plt.plot(y_train, rounded, 'ro', markersize=2)

predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]

plt.plot(y_test, rounded, 'bo', markersize=2)

plt.legend('Seen/Training Images','Validation/Unseen Images')
plt.title("Performance of KerasTuner-optimised model")
plt.xlabel("Expected Tensile Yield Strength (MPa)")
plt.ylabel("Predicted Tensile Yield Strength (MPa)")
plt.show()

if (input("Save these outputs? (y/n)\n")) == 'y':
    filename = input("What do you want to call the file?\n")
    results = pd.DataFrame()
    results['Image'], results['Expected'], results['Predicted'] = [testimages, y_test, rounded]
    results.to_csv("ML/Results/"+filename+'.csv', index=False)

pass