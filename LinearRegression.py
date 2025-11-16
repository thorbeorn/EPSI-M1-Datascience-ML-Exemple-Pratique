from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import ssl
import certifi
import requests

try:
    X, y = fetch_california_housing(return_X_y=True, download_if_missing=False)
except:
    import requests
    ssl._create_default_https_context = ssl._create_unverified_context
    X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression().fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Coefficient R² :", r2)
print("MSE :", mse)
print("Coefficients du modèle :", model.coef_)
print("Intercept :", model.intercept_)