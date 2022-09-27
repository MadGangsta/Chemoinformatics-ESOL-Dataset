import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from rdkit.Chem import PandasTools
from rdkit.Chem import Crippen
from rdkit import Chem

# Загрузим датасет ESOL
DATA_PATH = 'data/'
df = pd.read_csv(DATA_PATH + 'delaney-processed.csv')
PandasTools.AddMoleculeColumnToFrame(df,'smiles','molec')

# Извлечем липофильность(LogP)
LogP_list=[]
for i in range(len(df[['molec']])):
    mol = Chem.Crippen.MolLogP(df.loc[i,'molec'])
    LogP_list.append(mol)
df['LogP'] = LogP_list

# Разделение датасета на тестовую и тренировочную выборку
train_df = df.sample(frac=.9, random_state=43)
test_df = df[~df.index.isin(train_df.index)]

independ = 'Number of Rotatable Bonds'
independ1 = 'Molecular Weight'
independ2 = 'LogP'
depend = 'measured log solubility in mols per litre' #зависимыйYYYY

train_y = np.asanyarray(train_df[[depend]])
train_X = np.asanyarray(train_df[[independ,independ1,independ2]])

# Кросс-валидация
best_score=0
best_param=0
score_arr=[]
param_arr=[]
for i in range(1,202, 25):
    der = RandomForestRegressor(max_depth=i)
    scores=cross_val_score(der, train_X, train_y.ravel(), cv=10)
    score_arr.append(scores.mean())
    param_arr.append(i)
    if best_score < scores.mean():
        best_score = scores.mean()
        best_param = i
print(best_param)
plt.figure(figsize=(8,5))
plt.scatter(param_arr,score_arr,color='black')
plt.plot(param_arr,score_arr,color='black')
plt.xlabel('Hyperparameter value')
plt.ylabel('Accuracy')
plt.grid()
plt.title('Cross-validation for Random Forest')
plt.show()

# Тренировка модели методом случайного леса
der = RandomForestRegressor(max_depth=6)
der.fit(train_X, train_y.ravel())
print(cross_val_score(der, train_X, train_y.ravel(), cv=5))
test_X = np.asanyarray(test_df[[independ,independ1,independ2]])
test_y = np.asanyarray(test_df[[depend]])
test_Y_ = der.predict(test_X)

# Результат
print(r2_score(test_y,test_Y_))

# Выводим график
plt.figure(figsize=(8,5))
plt.scatter(test_y,test_Y_,color='black')
plt.xlabel('actual')
plt.ylabel('predicted')
plt.grid()
plt.title('Random Forest')

plt.show()