'''
read file +
implement results 
teach
make linear data model
calculate error
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import statsmodels .api as sm

url = "C://Users/1358365/Desktop/lab4/train.csv"

dataset=pd.read_csv(url, header=0)

#exploring dataset
print(dataset.shape)
#print(dataset.head(20))
#print(dataset.describe())

#show data
#dataset.plot(kind='box', subplots=True, layout=(10,10), sharex=False, sharey=False)
dataset.hist()
plt.show()

'''
price = dataset[dataset.columns[80]]
#print(price)

for i in range(80):
    df = dataset[dataset.columns[i]]
    print(df)
    result = df.join(price, 'left', on='Name')
   ''' 
