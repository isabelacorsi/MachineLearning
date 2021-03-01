import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 5:6].values
y = base.iloc[:, 2].values

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

plt.scatter(X_treinamento, y_treinamento)
plt.plot(X_treinamento, regressor.predict(X_treinamento), color = 'red')

previsoes = regressor.predict(X_teste)

resultado = abs(y_teste - previsoes)
resultado.mean()

mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)

plt.scatter(X_teste, y_teste)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')

print(regressor.score(X_teste, y_teste))
