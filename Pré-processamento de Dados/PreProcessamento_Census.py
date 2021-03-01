import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('census.csv') #carregando a base de dados

#Separação das colunas de previsores e de classe
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

#transforma valores nomiais em numéricos
labelencoder_previsores = LabelEncoder()

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])],
                                   remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()


labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

#padronização dos dados
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

print(previsores)