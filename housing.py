# basics
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


# load dataset
data = fetch_california_housing(as_frame=True)


# combine features + target into one dataframe
df = pd.concat(
    [data.data, data.target.rename("MedHouseVal")],
    axis=1
)

print(df.head())


# train/test split
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# model training
model = LinearRegression()

model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)


# metrics
mae = mean_absolute_error(y_test, y_pred)

# RMSE = Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R2 Score: {r2:.3f}")


# plot
plt.scatter(y_test, y_pred, alpha=0.4)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")

# ideal prediction line
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)]
)

plt.show()

