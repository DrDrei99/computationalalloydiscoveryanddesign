import pandas as pd

df2 = pd.read_csv("temp.csv")
newdf = pd.DataFrame(columns = ['1','2','3','4'])
images = []
mat = []
prop = []
src = []
df = df2.loc[df2['Image'].astype('str').str.isnumeric()]

print(df)
for name in df['Name']:
    for num in range(0,df.loc[df['Name'] == name, 'Image'].astype(int).item()):
        images.append(name + '-' + str(num+1    ) + '.png')
        mat.append(name)
        prop.append(df.loc[df['Name']==name, 'Tensile Strength (Yield)'].item())
        src.append(df.loc[df['Name']==name, 'Prop Source'].item())

newdf['1'] = images
newdf['2'] = mat
newdf['3'] = prop
newdf['4'] = src

newdf.to_csv("temp2.csv")