
# Introduction

This notebook contains an attempt to solve the [House Prices](https://www.kaggle.com/vikrishnan/house-sales-price-using-regression) regression challenge at Kaggle. If you are going to run this notebook, you should also download the challenge data into a directory `Data/` in the same directory as this notebook.

This work was motivated by the EEL860 course at UFRJ during the first semester of 2018.

**Author**: Pedro Boueke

**Source**: https://github.com/pboueke/kaggle-house-prices

## Understanding the data 

First things first. We need to understand our data. Lets start by taking a look at the training data.


```python
# Importing packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from matplotlib import pyplot
from sklearn.preprocessing import Imputer
import warnings
%matplotlib inline
```


```python
# Loading data
raw_train = pd.read_csv("Data/train.csv")
raw_test = pd.read_csv("Data/test.csv")
# Describing the data
raw_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
raw_train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>...</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1201.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1452.000000</td>
      <td>1460.000000</td>
      <td>...</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
      <td>1460.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>730.500000</td>
      <td>56.897260</td>
      <td>70.049958</td>
      <td>10516.828082</td>
      <td>6.099315</td>
      <td>5.575342</td>
      <td>1971.267808</td>
      <td>1984.865753</td>
      <td>103.685262</td>
      <td>443.639726</td>
      <td>...</td>
      <td>94.244521</td>
      <td>46.660274</td>
      <td>21.954110</td>
      <td>3.409589</td>
      <td>15.060959</td>
      <td>2.758904</td>
      <td>43.489041</td>
      <td>6.321918</td>
      <td>2007.815753</td>
      <td>180921.195890</td>
    </tr>
    <tr>
      <th>std</th>
      <td>421.610009</td>
      <td>42.300571</td>
      <td>24.284752</td>
      <td>9981.264932</td>
      <td>1.382997</td>
      <td>1.112799</td>
      <td>30.202904</td>
      <td>20.645407</td>
      <td>181.066207</td>
      <td>456.098091</td>
      <td>...</td>
      <td>125.338794</td>
      <td>66.256028</td>
      <td>61.119149</td>
      <td>29.317331</td>
      <td>55.757415</td>
      <td>40.177307</td>
      <td>496.123024</td>
      <td>2.703626</td>
      <td>1.328095</td>
      <td>79442.502883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>34900.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>365.750000</td>
      <td>20.000000</td>
      <td>59.000000</td>
      <td>7553.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1967.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2007.000000</td>
      <td>129975.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>730.500000</td>
      <td>50.000000</td>
      <td>69.000000</td>
      <td>9478.500000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1994.000000</td>
      <td>0.000000</td>
      <td>383.500000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>163000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1095.250000</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11601.500000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2000.000000</td>
      <td>2004.000000</td>
      <td>166.000000</td>
      <td>712.250000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>68.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>214000.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1460.000000</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>5644.000000</td>
      <td>...</td>
      <td>857.000000</td>
      <td>547.000000</td>
      <td>552.000000</td>
      <td>508.000000</td>
      <td>480.000000</td>
      <td>738.000000</td>
      <td>15500.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 38 columns</p>
</div>



**Good**. The value we are trying to predict is the `SalePrice`, the last column of the training data. We can see that it is not available in the test data. Now we should plot the data to see if there are any obvious correlations between the attributes, but before that we will remove the ID column and save it elsewhere. And fill missing values. And apply a hashing trick.


```python
# Saving the ids
train_id = raw_train['Id']
test_id = raw_test['Id']
raw_train.drop("Id", axis = 1, inplace = True)
raw_test.drop("Id", axis = 1, inplace = True)
```


```python
# Hashing trick
val_hash = {}
train = pd.DataFrame()
test = pd.DataFrame()
for dataset in [(raw_train, train), (raw_test, test)]:
    for col in dataset[0].axes[1]: 
        if dataset[0][col].dtype == 'object': # string columns
            if col not in val_hash:
                val_hash[col] = {}
            cid = 0
            for i,val in enumerate(dataset[0][col]):
                if val not in val_hash[col]:
                    val_hash[col][val] = cid
                    cid += 1
            dataset[1][col] = [val_hash[col][x] for x in dataset[0][col]]
        else:
            dataset[1][col] = dataset[0][col]        
```


```python
# Replacing null values with the mean for the column
train = train.fillna(train.mean())
test = test.fillna(test.mean())
```

Starting with an histogram of the SalePrice variable:


```python
sns.distplot(train['SalePrice']);
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
plt.show()
```


![png](output_9_0.png)



    <Figure size 1440x216 with 0 Axes>


Now a correlation matrix with all the variables:


```python
corrmat = train.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corrmat, vmax=.8, square=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f474a48ea58>




![png](output_11_1.png)


Looking specifically at the SalePrice variable, we can see some very interesting features. The variable seems to be well correlated with some other variables such as `OverallQual`, `GrLivArea` and a few others. Looking at the `data_description.txt` file, we see that:

```
OverallQual: Rates the overall material and finish of the house
       10	Very Excellent
       [...]
       1	Very Poor
       
[...]

GrLivArea: Above grade (ground) living area square feet
```

This is a great start, but we must still look further into the data. Lets rank the other variables by their correlation with `SalePrice` and remove the least correlated by an arbitrary `0.5` correlation value:


```python
most_corr = pd.DataFrame(corrmat['SalePrice'])
most_corr = most_corr[most_corr.SalePrice > 0.5]
most_corr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>OverallQual</th>
      <td>0.790982</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>0.522897</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>0.507101</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>0.613581</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>0.605852</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>0.708624</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>0.560664</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>0.533723</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>0.640409</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>0.623431</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



So we have **10** most correlated features - all of them numerical! Lets plot them aggregating the price values in ranges:


```python
new_set = pd.DataFrame(train)
bucket_size = 100000
for feature in train.axes[1]:
    if feature not in most_corr.axes[0]:
        new_set.drop(feature, axis = 1, inplace = True)
        
new_set.drop("SalePrice", axis = 1, inplace = True)
SalePriceRange = pd.DataFrame({"SalePriceRange":[str((int(x) // bucket_size) * bucket_size) for x in train["SalePrice"]]})
new_set = pd.concat([new_set, SalePriceRange], axis=1, sort=False)
#range_data = pd.DataFrame({'SalePriceRange': [str((int(x) // bucket_size) * bucket_size) for x in train["SalePrice"]]})
#new_set.join(range_data)
new_set["SalePriceRange"].describe()

sns.countplot(new_set["SalePriceRange"]);
plt.ylabel('Count')
plt.title('SalePriceRange distribution')
ax = plt.gca()
ax.set_xticklabels(ax.get_xticklabels(),rotation = (-45), fontsize = 10, va='top', ha='left')
fig = plt.figure()
plt.show()
```


![png](output_15_0.png)



    <Figure size 1440x216 with 0 Axes>



```python
axes_vars = [x for x in new_set.axes[1] if x != "SalePriceRange"]
g = sns.PairGrid(new_set, vars=axes_vars, hue="SalePriceRange")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend();

```


![png](output_16_0.png)


We can play around with the bucket size for this plot for a while, as I did, but even now we can see that there really are some combinations of features with waht seems to be a close to linear distinction of ranges. For example, at the plot [7,4]. Before we start modeling our problem, we must make some considerations about missing data and outliers. For this initial approach I will accept NA values as categoriacal data in all cases - which, for most features, is true, and will ignopre outliers. We may need to revise these decisions later.

For now, lets prepare the training/test data and a bunch of the models we will be comparing.


```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from pylab import rcParams
rcParams['figure.figsize'] = 5, 10
plt.rcParams["figure.figsize"] = (20,3)
warnings.filterwarnings('ignore')

```

Instantiating multiple regressors:


```python
models = []
models.append(['LR', LinearRegression()])
models.append(['RD', Ridge()])
models.append(['BR', BayesianRidge(compute_score=True)])
models.append(['LASSO', Lasso(alpha=0.015)])
models.append(['LARS', Lars(positive=True)])
models.append(['OMP', OrthogonalMatchingPursuit()])
models.append(['LG', LinearRegression()])
models.append(['EN', ElasticNet(max_iter=30)])
models.append(['KNN', KNeighborsRegressor()])
models.append(['CART', DecisionTreeRegressor()])
models.append(['SVR', SVR()])
models.append(['XGB', xgb.XGBRegressor()])
models.append(['GBR', GradientBoostingRegressor()])
models.append(['LGB', lgb.LGBMRegressor()])

# two groups of feature: those most correlated to our target and all of them
features_all = train.loc[:, train.columns != 'SalePrice'].values
features_mc = new_set.loc[:, new_set.columns != 'SalePriceRange'].values
targets = train.loc[:, train.columns == 'SalePrice'].values.flatten()
```

Lets also try increasing the dataset in as many dimensions we can in order to increase the number of features with binary values.


```python
new_features = pd.DataFrame()
new_test = pd.DataFrame()
new_cols = {}
threshold = 25
counter = 0
# generate the new columns
for col in train.axes[1]:
    if col == 'SalePrice':
        continue
    if len(np.unique(train[col])) >= threshold:
        new_features[col] = train[col]
    else:
        new_cols[col] = {}
        for val in np.unique(train[col]):
            new_cols[col][val] = []
        for val in train[col]:
            for k,v in new_cols[col].items():
                if k == val:
                    v.append(0)
                else:
                    v.append(1)
        for k,v in new_cols[col].items():
            new_features[col+str(k)] = v
            counter += 1

# generate new test set
for col in test.axes[1]:
    if col in new_cols:
        for k in new_cols[col].keys():
            new_test[col+str(k)] = [(1 if x==k else 0) for x in test[col]]
    else:
        new_test[col] = test[col]
            
print("Added " + str(counter) + " new columns")
```

    Added 414 new columns



```python
metric = "neg_mean_squared_error" 
seed = None
test_size = 0.20
num_folds = 10
results = []
names = []
for i,features in enumerate([features_all, features_mc, new_features.values]):
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        res = cross_val_score(model, features, targets, cv=kfold, scoring=metric)
        res = res[abs(res - np.mean(res)) < 3 * np.std(res)]
        results.append(res)
        names.append(name + " " + str(i))
        print("%s %s: %f (%f)" % (name, str(i), np.sqrt(-res.mean()), res.std()))
```

    LR 0: 38352.369385 (1404253626.348016)
    RD 0: 36866.777175 (1152176813.274163)
    BR 0: 35531.102110 (1022034391.720950)
    LASSO 0: 38351.597016 (1404098488.652683)
    LARS 0: 41231.139199 (1696274475.944178)
    OMP 0: 38407.835646 (835328555.104638)
    LG 0: 38352.369385 (1404253626.348016)
    EN 0: 36003.106122 (1073285627.088135)
    KNN 0: 46510.819297 (855416561.288784)
    CART 0: 40523.756501 (1013016309.198936)
    SVR 0: 81443.384768 (1567149473.726038)
    XGB 0: 27464.921833 (294518899.215236)
    GBR 0: 26696.178931 (283280233.758811)
    LGB 0: 29519.298201 (391518319.716425)
    LR 1: 38966.549400 (906998623.180888)
    RD 1: 38964.636491 (907224436.979261)
    BR 1: 38939.259520 (914235733.640570)
    LASSO 1: 38966.548017 (906999397.569994)
    LARS 1: 38900.656697 (879545469.734735)
    OMP 1: 48695.594606 (640623241.185150)
    LG 1: 38966.549400 (906998623.180888)
    EN 1: 39498.705692 (1007915219.688071)
    KNN 1: 42241.925085 (748785573.354603)
    CART 1: 41131.289108 (642558702.584372)
    SVR 1: 81443.355897 (1567148890.967785)
    XGB 1: 29025.258146 (287426929.360501)
    GBR 1: 29766.522197 (428379801.849960)
    LGB 1: 32801.505752 (424745912.403068)
    LR 2: 140543214316.266144 (39818053860518130614272.000000)
    RD 2: 35059.462904 (1090169971.994500)
    BR 2: 33535.897884 (930936927.054557)
    LASSO 2: 37854.870421 (1200803428.854550)
    LARS 2: 40228891089686064427995303438378099638413348044800.000000 (4853143259254847281864495433111857051388856722545152817534087914909449843244751048103948717708541952.000000)
    OMP 2: 34612.323309 (1094214333.693496)
    LG 2: 140543214316.266144 (39818053860518130614272.000000)
    EN 2: 37012.351443 (1179113447.651754)
    KNN 2: 46425.846355 (878983661.807153)
    CART 2: 47248.387201 (747871222.031237)
    SVR 2: 81443.382451 (1567149481.499020)
    XGB 2: 27513.991075 (302131879.304635)
    GBR 2: 28238.534310 (416468441.021774)
    LGB 2: 29217.859717 (387724734.836107)



```python
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
bp_data = []
bp_labels = []
for i,name in enumerate(names):
    if name not in ["LARS 2", "LR 2", "LG 2"]:
        bp_data.append(results[i])
        bp_labels.append(names[i])
pyplot.boxplot(bp_data)
ax.set_xticklabels(bp_labels,rotation = (-45), fontsize = 10, va='top', ha='left')
pyplot.show()
```


![png](output_24_0.png)


We have seen that using only the most correlated features is not helpfull. Lets try againg scaling the data.


```python
pipeline = []
for model in models:
    pipeline.append([model[0], Pipeline([('Scaler', StandardScaler()),(model[0], model[1])])])
```


```python
results = []
names = []
for i,features in enumerate([features_all, new_features.values]):
    for name, model in pipeline:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        res = cross_val_score(model, features, targets, cv=kfold, scoring=metric)
        res = res[abs(res - np.mean(res)) < 2 * np.std(res)]
        results.append(res)
        names.append(name + " " + str(i))
        print("%s %s: %f (%f)" % (name, str(i), np.sqrt(-res.mean()), res.std()))
```

    LR 0: 39354.191498 (1459519548.634823)
    RD 0: 31943.531319 (424930763.429740)
    BR 0: 31426.199850 (415472498.544121)
    LASSO 0: 31978.338238 (425935677.101571)
    LARS 0: 34126.822556 (575230045.012155)
    OMP 0: 34931.807276 (354129500.829992)
    LG 0: 39354.191498 (1459519548.634823)
    EN 0: 31368.686493 (411463794.105994)
    KNN 0: 36642.047741 (416530700.353919)
    CART 0: 37845.474729 (400595276.342143)
    SVR 0: 81411.387692 (1567195021.958897)
    XGB 0: 25934.093280 (173362519.310129)
    GBR 0: 25200.570441 (172140283.743905)
    LGB 0: 27632.759182 (326568527.351541)
    LR 1: 318806296623811520.000000 (75992337523169860594883681146896384.000000)
    RD 1: 32791.760131 (634102522.764406)
    BR 1: 30183.721142 (507684374.502673)
    LASSO 1: 33288.752040 (633403989.631268)
    LARS 1: 81298378030265605360447444092834545664.000000 (16579671312775919695340429572498234020902439672751080666046738124789528920064.000000)
    OMP 1: 29832.660080 (617691618.835374)
    LG 1: 318806296623811520.000000 (75992337523169860594883681146896384.000000)
    EN 1: 28768.409449 (399998840.269236)
    KNN 1: 38237.334261 (475138001.272899)
    CART 1: 44931.730852 (515320552.995535)
    SVR 1: 81423.999192 (1567243273.587633)
    XGB 1: 25941.174776 (174929984.102358)
    GBR 1: 25553.147108 (156575459.631939)
    LGB 1: 27747.880838 (307728897.610874)



```python
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
bp_data = []
bp_labels = []
for i,name in enumerate(names):
    if name not in ["LARS 1", "LR 1", "LG 1", "SVR 1", "SVR 0"]:
        bp_data.append(results[i])
        bp_labels.append(names[i])
pyplot.boxplot(bp_data)
ax.set_xticklabels(bp_labels,rotation = (-45), fontsize = 10, va='top', ha='left')
pyplot.show()
```


![png](output_28_0.png)


Consistently our best option is using Gradient Boosting regressors without the standardization process and with all the original hashed features. Lets now try playing around with the parameters of each regressor.


```python
gradient_models = []
gradient_models.append(['XGB', xgb.XGBRegressor(colsample_bytree=0.7, gamma=0.0, 
                                   learning_rate=0.025, max_depth=6, 
                                   min_child_weight=1.5, n_estimators=7200,
                                   reg_alpha=0.9, reg_lambda=0.6,
                                   subsample=0.3, silent=1)])
gradient_models.append(['GBR', GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =7)])
gradient_models.append(['LGB', lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                   learning_rate=0.05, n_estimators=720,
                                   max_bin = 55, bagging_fraction = 0.8,
                                   bagging_freq = 5, feature_fraction = 0.2319,
                                   feature_fraction_seed=18, bagging_seed=18,
                                   min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)])

results = []
names = []
scaler = StandardScaler().fit(features_all)
rescaled_features = scaler.transform(features_all)
scaler = StandardScaler().fit(new_features.values)
rescaled_new_features = scaler.transform(new_features.values)

for i,rescaled in enumerate([features_all]):#, new_features.values, rescaled_features, rescaled_new_features]):
    for name, model in gradient_models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        res = cross_val_score(model, rescaled, targets, cv=kfold, scoring=metric)
        res = res[abs(res - np.mean(res)) < 3 * np.std(res)]
        results.append(res)
        names.append(name + " " + str(i))
        print("%s %s: %f (%f)" % (name, str(i), np.sqrt(-res.mean()), res.std()))
```

    XGB 0: 26673.456274 (351821811.012833)
    GBR 0: 27401.600794 (465460835.339232)
    LGB 0: 27546.749755 (386812380.622776)


We are going to use the average prediction of the selected regressors to calculate our final results


```python
all_records = new_features.append(new_test)
scaler = StandardScaler().fit(final_train_set)
rescaled_features = scaler.transform(new_features)
rescatled_test = scaler.transform(new_test.values)

x = rescaled_features
y = targets
predictions = []
final = []
for name, model in gradient_models:
    regressor = model.fit(x,y)
    predictions.append(regressor.predict(rescatled_test))
    
final = [sum([x[i] for x in predictions])/len(predictions) for i in range(len(test))]
```

And finally generating the submission


```python
sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = final
sub.to_csv('Data/submission.csv',index=False)
```

## Conclusion

Once submitting a few times, the best score I managed was 0.12496, which puts me in the 1239th position as of 2018-07-26. This was obtained  using the results of the hashing trick over the extended and reescaled dataset.

There is a lot of room for improvement, specially if we look into the most correlated features. For the future, we should look into normalizing the distributions of such features.
