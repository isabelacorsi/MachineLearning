import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('credit_data.csv')

#Alguns valores de Age estão negativos
# preencher os valores com a média
base.mean()
base['age'].mean()
base['age'][base.age > 0].mean()
base.loc[base.age < 0, 'age'] = 40.92

pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])


scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)














