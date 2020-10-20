from sklearn import datasets
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\risha\Downloads\Iris.csv")

df.head()
print('Data read successfully')

df.drop(['Id'],axis=1,inplace=True)
print(df.head())

print(df['Species'].value_counts('Normalize=1'))

y = df.iloc[:,-1]
x = df.iloc[:,:-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)
print(x_train.head())

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5,criterion='gini')
print(model.fit(x_train,y_train))

y_test_pred = model.predict(x_test)
dframe = pd.DataFrame({'Actual':y_test, 'Predicted':y_test_pred})
print(dframe)

plt.figure(figsize=(15,10))
tree.plot_tree(model,filled=True)
plt.title('Decision Tree')
plt.show()

