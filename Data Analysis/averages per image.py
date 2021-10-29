import pandas as pd
import numpy as np

dataset = pd.read_csv("final_dataset.csv")
files = [#'guessed-whole-MCCV-8020',
        #'guessed-whole-MCCV-9010',
        #'guessed-magnesium-MCCV-8020',
        #'guessed-magnesium-MCCV-9010',
        #'KT-whole-MCCV-8020',
        #'KT-whole-MCCV-9010',
        #'KT-magnesium-MCCV-8020',
        #'KT-magnesium-MCCV-9010',
        #"KT-whole-MCCV-9010-100run",
        #"KT-whole-MCCV-9010-100run-titanium",
        "KT-whole-MCCV-9010-200run"]

for filename in files:
    results = pd.read_csv("ML/Results/"+filename+".csv")

    imagenames = list(results["Image"].values)

    names = [x[0:x.rfind("c")] for x in imagenames]

    results['Baseimage'] = names 

    nameunique = list(set(names))

    expected = [np.average(results.loc[results['Baseimage']==name]['Expected'].values) for name in nameunique]
    averages = [np.average(results.loc[results['Baseimage']==name]['Predicted'].values) for name in nameunique]
    source = [results.loc[results['Baseimage']==name]['Source'].values[0] for name in list(set(names))]

    average_results = pd.DataFrame()
    average_results['Source'] = source
    average_results['Image Name'] = nameunique
    average_results['Expected'] = expected
    average_results['Average Prediction'] = averages
    average_results.to_csv("ML/Results/Averages/"+filename+"imageaverages.csv", index=None)

pass