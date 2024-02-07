import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"D:/Data Scientist/Supervised Learning/Regression/Simple Linear Regression/Simple Linear Regression/datasets/calories_consumed.csv")

from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user="root",# user
                               pw="9130439933", # passwrd
                               db="calories"))

data.to_sql('data',con = engine, if_exists = 'replace', index = False)

sql = "SELECT * FROM data";

df = pd.read_sql_query(sql,engine)

df.head()
df.describe()
df.info()

df.value_counts()

df.columns = ['weight','calories_consumed']

X = pd.DataFrame(df['weight'])
y = pd.DataFrame(df['calories_consumed'])

numeric_features = ['weight']

df.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8))

import pandas as pd # deals with data frame        # for Data Manipulation"
import numpy as np  # deals with numerical values  # for Mathematical calculations"
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer
from sklearn.linear_model import LinearRegression
 

winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = numeric_features)
winsor

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])
outlier_pipeline = Pipeline(steps = [('winsor', winsor)])
num_pipeline
outlier_pipeline


preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])
print(preprocessor)

preprocessor1 = ColumnTransformer(transformers = [('wins', outlier_pipeline, numeric_features)])
print(preprocessor1)

impute_data = preprocessor.fit(X)
df['weight'] = pd.DataFrame(impute_data.transform(X))

X2 = pd.DataFrame(df['weight'])
winz_data = preprocessor1.fit(X2)

df['weight'] = pd.DataFrame(winz_data.transform(X))

# Save the data preprocessing pipelines
joblib.dump(impute_data, 'meanimpute')

joblib.dump(winz_data, 'winzor')

# # Bivariate Analysis
# Scatter plot
plt.scatter(x = df['weight'], y = df['calories_consumed']) 

np.corrcoef(df.weight, df.calories_consumed)

# # Linear Regression using statsmodels package
# Simple Linear Regression
model = smf.ols('calories_consumed ~ weight', data = df).fit()

model.summary()

pred1 = model.predict(pd.DataFrame(df['weight']))

pred1

# Regression Line
plt.scatter(df.weight, df.calories_consumed)
plt.plot(df.weight, pred1, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()


# Error calculation (error = AV - PV)
res1 = df.calories_consumed - pred1

print(np.mean(res1))

res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


plt.scatter(x = np.log(df['weight']), y = df['calories_consumed'], color = 'brown')
np.corrcoef(np.log(df.weight), df.calories_consumed) #correlation

model2 = smf.ols('calories_consumed ~ np.log(weight)', data = df).fit()
model2.summary()


pred2 = model2.predict(pd.DataFrame(df['weight']))

# Regression Line
plt.scatter(np.log(df.weight), df.calories_consumed)
plt.plot(np.log(df.weight), pred2, "r")
plt.legend(['Observed data', 'Predicted line'])
plt.show()

# Error calculation
res2 = df.calories_consumed- pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

# # Evaluate the best model
# Data Split
train, test = train_test_split(df, test_size = 0.2, random_state = 0)

plt.scatter(train.weight, train.calories_consumed)

plt.figure(2)
plt.scatter(test.weight, test.calories_consumed)

# Fit the best model on train data
finalmodel = smf.ols('calories_consumed ~ weight', data =train).fit()

# Predict on test data
test_pred = finalmodel.predict(test)
pred_test_calories_consumed = test_pred

# Model Evaluation on Test data
test_res = test.calories_consumed - pred_test_calories_consumed
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)

test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_calories_consumed = train_pred
pred_train_calories_consumed

# Model Evaluation on train data
train_res = train.calories_consumed - pred_train_calories_consumed
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)

train_rmse


pickle.dump(finalmodel, open('poly_model.pkl', 'wb'))
