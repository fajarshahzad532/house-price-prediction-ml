import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#data set

data = {
    "size": [
        500, 600, 700, 800, 900,
        1000, 1100, 1200, 1300, 1400,
        1500, 1600, 1700, 1800, 1900,
        2000, 2100, 2200, 2300, 2400,
        2500, 2600, 2700, 2800, 2900,
        3000, 3100, 3200, 3300, 3400,
        3500, 3600, 3700, 3800, 3900,
        4000, 4100, 4200, 4300, 4400,
        4500, 4600, 4700, 4800, 4900
    ],

    "price": [
        2, 2.5, 3, 3.5, 4,
        4.2, 4.5, 5, 5.5, 6,
        6.2, 6.5, 7, 7.2, 7.5,
        8, 8.3, 8.7, 9, 9.5,
        9.8, 10.2, 10.5, 11, 11.3,
        11.7, 12, 12.5, 12.8, 13.2,
        13.5, 14, 14.3, 14.7, 15,
        15.5, 15.8, 16.2, 16.5, 17,
        17.5, 18, 18.5, 19, 19.5
    ]
}

df = pd.DataFrame(data)

#splitting features and target

X = df[["size"]]
y = df["price"]

#splitting up of data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#model training

model = LinearRegression()
model.fit(X_train, y_train)

#calculating mean suare error of training data
y_pred_train = model.predict(X_train)
train_mse = mean_squared_error(y_train, y_pred_train)
print("train MSE:", train_mse)


#prediction
y_pred_test = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print("test MSE:", test_mse)