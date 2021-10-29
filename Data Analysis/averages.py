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

    names = [dataset.loc[dataset["Image"] == imagename]["Name"].values[0] for imagename in results["Image"].values]

    results['Name'] = names 

    nameunique = list(set(names))

    expected = [np.average(results.loc[results['Name']==name]['Expected'].values) for name in list(set(names))]
    averages = [np.average(results.loc[results['Name']==name]['Predicted'].values) for name in list(set(names))]
    source = [results.loc[results['Name']==name]['Source'].values[0] for name in list(set(names))]

    average_results = pd.DataFrame()
    average_results['Source'] = source
    average_results['Alloy Name'] = nameunique
    average_results['Expected'] = expected
    average_results['Average Prediction'] = averages
    average_results.to_csv("ML/Results/Averages/"+filename+"averages.csv", index=None)

pass