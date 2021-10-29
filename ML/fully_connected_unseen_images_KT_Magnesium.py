import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
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
    
    for i in range(hp.Int("num_layers",min_value = 1, max_value = 3, step = 1)):
        model.add(Dense(
            units=hp.Int("units_"+str(i), min_value=32, max_value = 512, step = 32),
            activation='relu'
        ))
    
    model.add(Dense(1, activation='relu'))
    model.compile(loss="mse",optimizer="adam", metrics=["mse"])
    
    return model


featurevectors = pd.read_csv("vggoutputs.csv")
dataset = pd.read_csv("final_dataset.csv")

dataset = dataset.loc[dataset['Source'] == 'Magnesium Dataset'][dataset.columns]

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
    objective = "val_mse",
    max_epochs=50,
    seed=69,
    directory = 'ML/Tuner',
    project_name = 'Research_project_Hyperband_Magnesium'
    , overwrite=True
)

tuner.search(X_train, y_train, epochs=60, callbacks=[EarlyStopping(monitor='val_mse', patience=3, mode='min')], validation_data = (X_test, y_test))

model2 = tuner.get_best_models(num_models=1)[0]

model2.save("ML/Tuner/optimal_model_trained_magnesium")

best_hp = tuner.get_best_hyperparameters()[0]
model1 = tuner.hypermodel.build(best_hp)

model1.save("ML/Tuner/optimal_model_untrained_magnesium")

#_, mse = model2.evaluate(X_train, y_train)
#print('Accuracy: %.2f' % (mse))

predictions = model2.predict(X_train)
train_predictions = [round(x[0]) for x in predictions]

predictions = model2.predict(X_test)
test_predictions = [round(x[0]) for x in predictions]

plt.plot(y_train, train_predictions, 'ro', markersize=2, label = 'Seen/Training Images')

plt.plot(y_test, test_predictions, 'bo', markersize=2, label = 'Validation/Unseen Images')

plt.legend('Seen/Training Images','Validation/Unseen Images')
plt.title("Performance of KerasTuner-optimised model on Magnesium Dataset")
plt.xlabel("Expected Tensile Yield Strength (MPa)")
plt.ylabel("Predicted Tensile Yield Strength (MPa)")
plt.show()

total_predictions = test_predictions + train_predictions
split = ['Test' for x in testimages] + ['Train' for x in trainimages]


if (input("Save these outputs? (y/n)\n")) == 'y':
    filename = input("What do you want to call this?\n")
    results = pd.DataFrame()
    results['Image'] = np.append(testimages, trainimages)
    results['Expected'] = np.append(y_test, y_train)
    results['Predicted'] = total_predictions
    results['Subset'] = split
    results.to_csv("ML/Results/"+filename+'.csv', index=False)

pass