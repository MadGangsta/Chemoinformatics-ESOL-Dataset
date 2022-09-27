import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Crippen
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Загрузим датасет ESOL
DATA_PATH = 'data/'
df = pd.read_csv(DATA_PATH + 'delaney-processed.csv')
PandasTools.AddMoleculeColumnToFrame(df,'smiles','molec')

# Извлечем липофильность(LogP)
LogP_list=[]
AP_list = []
for i in range(len(df[['molec']])):
    mol = Chem.Crippen.MolLogP(df.loc[i,'molec'])
    LogP_list.append(mol)
df['LogP'] = LogP_list

# Разделение датасета на тестовую и тренировочную выборку
train_df = df.sample(frac=.8, random_state=43)
test_df = df[~df.index.isin(train_df.index)]

independ = 'Number of Rotatable Bonds'
independ1 = 'Molecular Weight'
independ2 = 'LogP'
depend = 'measured log solubility in mols per litre'

# Тренировка мультилинейной регрессии
regr = linear_model.LinearRegression()
train_y = np.asanyarray(train_df[[depend]])
train_X = np.asanyarray(train_df[[independ,independ1,independ2]])
regr.fit(train_X,train_y)
test_X = np.asanyarray(test_df[[independ,independ1,independ2]])
test_y = np.asanyarray(test_df[[depend]])
test_Y_ = regr.predict(test_X)

# Тренировка решающих деревьев
train_y = np.asanyarray(train_df[[depend]])
train_X = np.asanyarray(train_df[[independ,independ1,independ2]])
der = DecisionTreeRegressor(max_depth=9)
der.fit(train_X, train_y.ravel())
test_X1 = np.asanyarray(test_df[[independ,independ1,independ2]])
test_y1 = np.asanyarray(test_df[[depend]])
test_Y1 = der.predict(test_X1)

# Тренировка модели методом случайного леса
der = RandomForestRegressor(max_depth=9)
der.fit(train_X, train_y.ravel())
test_X2 = np.asanyarray(test_df[[independ,independ1,independ2]])
test_y2 = np.asanyarray(test_df[[depend]])
test_Y2 = der.predict(test_X2)

# Тренировка модели методом градиентного бустинга
regr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,loss='ls')
train_y = np.asanyarray(train_df[[depend]])
train_X = np.asanyarray(train_df[[independ,independ1,independ2]])
regr.fit(train_X, train_y.ravel())
test_X3 = np.asanyarray(test_df[[independ,independ1,independ2]])
test_y3 = np.asanyarray(test_df[[depend]])
test_Y3 = regr.predict(test_X3)

# Результат
print(r2_score(test_y,test_Y_))
print(r2_score(test_y1,test_Y1))
print(r2_score(test_y2,test_Y2))
print(r2_score(test_y3,test_Y3))

# Выводим график
plt.figure(figsize=(8,5))
plt.scatter(test_y,test_Y_)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.grid()

plt.show()

