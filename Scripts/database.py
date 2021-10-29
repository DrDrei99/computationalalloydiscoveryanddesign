import os
import pandas as pd

folder = 'Ti-6Al-2Sn-4Zr-2Mo-0.1Si'

df = pd.DataFrame(columns=['Path'])

counter = 0
for image in os.listdir(folder):
    path = folder + '/' + image
    df.loc[counter] = [path]
    counter = counter + 1

df.to_csv('titanium.csv')