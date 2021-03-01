import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

base = pd.read_csv('house_prices.csv')

X = base.iloc[:, 3:19].values
y = base.iloc[:, 2].values

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
regressor = LinearRegression()
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

previsoes = regressor.predict(X_teste)

mae = mean_absolute_error(y_teste, previsoes)

print(regressor.score(X_teste, y_teste))

regressor.intercept_
len(regressor.coef_)


