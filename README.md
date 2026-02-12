# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for the marks by using the representation of the graph.
6.Hence we obtained the linear regression for the given dataset. 


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VARDHANSAI I
RegisterNumber:  25015695
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\acer\Downloads\student_scores.csv")

print(df.head(10))

plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()


X = df[['Hours']]   
y = df['Scores']    

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, Y_train)

single_pred = lr.predict(X_test.iloc[0].values.reshape(1, 1))
print("Single Prediction:", single_pred)

plt.scatter(df['Hours'], df['Scores'])
plt.plot(X_train, lr.predict(X_train), color='red')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Regression Line')
plt.show()

print("Coefficient:", lr.coef_)
print("Intercept:", lr.intercept_)

y_pred = lr.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)
```

## Output:
<img width="1032" height="900" alt="Screenshot 2026-02-12 171055" src="https://github.com/user-attachments/assets/8608f081-b1ad-470c-8768-2a92f0be0c25" />
<img width="1104" height="854" alt="Screenshot 2026-02-12 171111" src="https://github.com/user-attachments/assets/f74c7d35-c45c-472b-b1e2-26b94c5fc5ef" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
