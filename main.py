import numpy as np     # linear algebra
import pandas as pd    # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt    # plotting data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns

sns.set(color_codes=True)
chicks = pd.read_csv('chickweight.csv')
print(chicks)
print(chicks.describe())
ax = sns.countplot(x="Diet", data=chicks)
for p in ax.patches:
    ax.annotate('{}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

sns.scatterplot(x='Time',y='weight', hue="Diet", size='Diet', data=chicks)
chicks_pivot = chicks.pivot_table(values="weight", index=['Chick', 'Diet'], columns='Time')
print(chicks_pivot)
chicks_pivot.isnull().any(axis=1).sum()
chicks_pivot = chicks_pivot.dropna()
chicks = chicks_pivot.stack().reset_index(name='weight')
g = sns.FacetGrid(chicks, col="Diet", margin_titles=True)
g.map(sns.regplot, "Time", "weight",fit_reg=False, x_jitter=.1)
sns.catplot(x="Time", y="weight", col="Diet", data=chicks, kind="box", col_wrap=2)
chicks_gb = chicks.groupby('Diet').agg(
    max_weight=('weight', max),
    min_weight=('weight', min),
    avg_weight=('weight', 'mean'),
    total_weight=('weight', sum),
    num_chicks=('Chick', 'count')
)

# y = WEIGHT -> target
y = chicks.weight
# Create X = ["Diet", "Time"] -> data
features = ["Diet", "Time"]
X = chicks[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.2)

# Specify Model and fit it
C_modelRegr =  LinearRegression().fit(train_X, train_y)
C_modelRidge = Ridge().fit(train_X, train_y)
C_modelLasso = Lasso().fit(train_X, train_y)
C_modelElasticNet = ElasticNet(alpha = 0.1).fit(train_X, train_y)

print("Linear Regression:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelRegr.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelRegr.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelRegr.score(val_X, val_y)))

print("Ridge Regression:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelRidge.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelRidge.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelRidge.score(val_X, val_y)))

print("Lasso Regression:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelLasso.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelLasso.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelLasso.score(val_X, val_y)))

print("Elastic Net:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelElasticNet.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelElasticNet.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelElasticNet.score(val_X, val_y)))

C_modelKNN = KNeighborsRegressor(n_neighbors = 25).fit(train_X, train_y)
print("KNeighbors Regressor:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelKNN.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelKNN.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelKNN.score(val_X, val_y)))

# Specify Model and fit it
C_modelDTreeRegr = DecisionTreeRegressor(max_leaf_nodes = 25, random_state=1).fit(train_X, train_y)
C_modelForest = RandomForestRegressor(n_estimators=70, max_depth=3, random_state=1).fit(train_X, train_y)
C_modelMLPRegr = MLPRegressor(solver = 'lbfgs', random_state = 1, hidden_layer_sizes =[10]).fit(train_X, train_y)

print("Decision Tree Regressor:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelDTreeRegr.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelDTreeRegr.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelDTreeRegr.score(val_X, val_y)))

print("Random Forest Regressor:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelForest.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelForest.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelForest.score(val_X, val_y)))

print("MLP Regressor:")
print("\tValidation MAE: {:,.0f}".format(mean_absolute_error(C_modelMLPRegr.predict(val_X), val_y)))
print("\tAccuracy on train data: {:.3f}".format(C_modelMLPRegr.score(train_X, train_y)))
print("\tAccuracy on test data: {:.3f}".format(C_modelMLPRegr.score(val_X, val_y)))


