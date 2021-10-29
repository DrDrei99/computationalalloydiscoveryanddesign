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

test_split_input = False
while test_split_input==False:
    split_mode = int(input('What train/test split will you use?\n 80:20 is "1" \n 90:10 is "2"\n Input Number:'))
    if not split_mode in (1,2):
            print("Not a valid option")
            continue
    test_split_input = True

if split_mode == 1:
    test_split = 0.2
    num_runs = 5
if split_mode == 2:
    test_split = 0.1
    num_runs = 10

model = sequential.Sequential()
model.add(InputLayer(input_shape=(513,)))
model.add(Dense(288, activation='relu'))
model.add(Dense(320, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss="mse",optimizer="adam")

epoch_num = 50
batch_num = 32

model.save("ML/Models/Guessed")

test_predictions = []
test_expected = []
split = []
total_test_images = []

for run in range(0,num_runs):
    model = load_model("ML/Models/Guessed")

    train, test = train_test_split(baseimages, test_size=test_split)

    trainimages = [image for image in imagenames if (image[0:image.rfind("Q")+2] in train)]
    testimages = [image for image in imagenames if (image[0:image.rfind("Q")+2] in test)]

    total_test_images = np.append(total_test_images, testimages)

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

    test_expected = np.append(test_expected, expected_strength_test)

    featureinputs_train = np.append(featureinputs_train, [scale_per_pixel_train], axis = 0)
    featureinputs_train = np.transpose(featureinputs_train)

    featureinputs_test = np.append(featureinputs_test, [scale_per_pixel_test], axis = 0)
    featureinputs_test = np.transpose(featureinputs_test)

    X_train = featureinputs_train
    y_train = expected_strength_train

    X_test = featureinputs_test
    y_test = expected_strength_test

    model.fit(X_train, y_train, epochs=epoch_num, batch_size=batch_num, validation_data=(X_test,y_test))

#    predictions1 = model.predict(X_train)
#    current_train_predictions = [round(x[0]) for x in predictions1]

    predictions2 = model.predict(X_test)
    current_test_predictions = [round(x[0]) for x in predictions2]
    test_predictions = np.append(test_predictions, current_test_predictions)

    run_no = [run for x in testimages]# + [run for x in trainimages]
    split = np.append(split, run_no)

#total_predictions = test_predictions + train_predictions


results = pd.DataFrame()
results['Image'] = total_test_images
results['Expected'] = test_expected
results['Predicted'] = test_predictions
results['Run No.'] = split
results['Source'] = [dataset.loc[dataset['Image'] == x]['Source'].values[0] for x in total_test_images]

#plt.plot(y_train, train_predictions, 'ro', markersize=2, label = 'Seen/Training Images')
plt.plot(test_expected, test_predictions, 'bo', markersize=2, label = 'Validation/Unseen Images')

plt.legend()
plt.title("Guessed architecture with " + str(batch_num) + " batch size and for " + str(epoch_num) + " epochs")
plt.xlabel("Expected Tensile Yield Strength (MPa)")
plt.ylabel("Predicted Tensile Yield Strength (MPa)")
plt.show()

if (input("Save these outputs? (y/n)\n")) == 'y':
    filename = input("What will you call this?\n")
    results.to_csv("ML/Results/"+filename+'.csv', index=False)

pass