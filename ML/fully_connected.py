import numpy as np
from keras.models import sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 


featurevectors = pd.read_csv("vggoutputs.csv")
dataset = pd.read_csv("final_dataset.csv")

featureinputs = featurevectors[featurevectors.columns].values

scalebar = dataset['Scale Length (px)'].values
scale = dataset['Scale'].values
scale_per_pixel = scale/scalebar

expected_strength = dataset['Tensile Strength (Yield)'].values

featureinputs = np.append(featureinputs, [scale_per_pixel], axis = 0)
featureinputs = np.transpose(featureinputs)

X = featureinputs
y = expected_strength

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = sequential.Sequential()
model.add(Dense(550, input_dim=513, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss="mean_squared_error",optimizer="adam", metrics=["mean_absolute_percentage_error"])

model.fit(X_train, y_train, epochs=1500, batch_size=1000)

_, accuracy = model.evaluate(X_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))

predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]

plt.plot(y_test, rounded, 'bo', markersize=2)
plt.title("Performance of evaluating test data")
plt.xlabel("Expected Tensile Yield Strength (MPa)")
plt.ylabel("Predicted Tensile Yield Strength (MPa)")
plt.pause(0)
pass