# %%
# import pyforest
from pyforest import*

import warnings 
warnings.filterwarnings('ignore')

# %%
# load the dataset
Bank_df = pd.read_csv('C:\\Users\\user\\Desktop\\Udemy\\Project-4\\Bank_note\\BankNote_Authentication+(1).csv')

# %%
# show first 5 rows 
Bank_df.head()

# %% [markdown]
# 0 means fake bank note, 1 means genunie bank note

# %%
# Check number of rows and columns 
Bank_df.shape

# %%
# info of the dataset 
Bank_df.info()

# %%
# Check the data types
Bank_df.dtypes

# %%
# Checking the null values 
Bank_df.isna().values.any()

# %%
Bank_df.isna().sum()

# %%
# Distribution of Target Variable 
Bank_df['class'].value_counts()

# %% [markdown]
# This is a fair distribution. Balanced dataset

# %% [markdown]
# ### **Visualization**

# %%
plt.style.use('fivethirtyeight')

# %%
# Distribution of Target Variable
plt.figure(figsize=(10,7))
plt.title('Distribution of "Class" Attribute')
sns.distplot(Bank_df['class'], color='red');

# %%
# Histogram multiple columns
Bank_df.hist(bins=20, figsize=(15,8), layout=(2, 3), color='blue');

# %%
# pairplot 
sns.pairplot(Bank_df, hue='class');

# %% [markdown]
# The correlation here is very poor. Some of the data points moderately correlated, Some of the data Negatively correlated, some of the data points are curve linear relationship.

# %% [markdown]
# ### **Preparing data to build our model**

# %%
Bank_df.head()

# %%
# Separate our data into dependent and independent 
X = Bank_df.drop('class', axis=1)
y = Bank_df['class']

# %%
# Split our data for training and testing 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# %%
# Scaling the data 
from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 
X = scaler.fit_transform(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# %%
X_train

# %%
X_test

# %% [markdown]
# ### **Logistic Regression**

# %%
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import accuracy_score

# %%
LogReg = LogisticRegression(solver='liblinear', random_state=42)
LogReg.fit(X_train, y_train)
accuracies = cross_val_score(estimator=LogReg, X=X_train, y=y_train, cv=10)
print('Accuracies:\n', accuracies)

# %%
# Print the mean of the accuracies
print("Mean Accuracy: ", accuracies.mean()*100)

# %%
# Test Predict
LogReg_pred = LogReg.predict(X_test)
LogReg_pred

# %%
# Confusion matrix
from sklearn import metrics 

cm = metrics.confusion_matrix(y_test, LogReg_pred, labels=[0,1])
df_cm = pd.DataFrame(cm, index=[i for i in [0,1]], columns=[i for i in ['Predicted 0', 'Predicted 1']])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm, annot=True, cmap='coolwarm', fmt='.2f', linecolor='k', linewidths=0.2)

# %% [markdown]
# ### **Support Vector Machine (SVM)**

# %%
from sklearn.svm import SVC 

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)
svm_accuracies = cross_val_score(estimator=svm_classifier, X=X_train, y=y_train, cv=10)
print("Accuracies:\n", svm_accuracies)

# %%
print("Mean Accuracy:", svm_accuracies.mean()*100)

# %%
# Test Pred
svm_classifier_pred = svm_classifier.predict(X_test)
svm_classifier_pred

# %%
# Confusion Matrix 

cm = metrics.confusion_matrix(y_test, svm_classifier_pred, labels=[0,1])
df_cm = pd.DataFrame(cm, index=[i for i in [0,1]], columns=[i for i in ['Predicted 0', 'Predicted 1']])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm, annot=True, cmap='coolwarm', fmt='.2f', linecolor='k', linewidths=0.2);

# %% [markdown]
# change the kernel= 'linear' to 'rbf'

# %%
rbf_classifier = SVC(kernel='rbf')
rbf_classifier.fit(X_train, y_train)
rbf_accuracies = cross_val_score(estimator=rbf_classifier, X=X_train, y=y_train, cv=10)
print("Accuracies:\n", rbf_accuracies)

# %% [markdown]
# This one is actually giving 100% good prediction.

# %%
print("Mean Accuracy:", rbf_accuracies.mean()*100)

# %%
# Test Pred
rbf_classifier_pred = rbf_classifier.predict(X_test)
rbf_classifier_pred

# %%
# Confusion Matrix

cm = metrics.confusion_matrix(y_test, rbf_classifier_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index=[i for i in [0, 1]], columns=[i for i in ['Predicted 0', 'Predicted 1']])
plt.figure(figsize=(7, 5))
sns.heatmap(df_cm, annot=True, cmap='coolwarm',fmt='.2f', linecolor='k', linewidths=0.2)

# %% [markdown]
# ### **Random Forest Classifier**

# %%
from sklearn.ensemble import RandomForestClassifier 

rfc = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
rfc.fit(X_train, y_train)
rfc_accuracies = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
print("Accuracies:\n",rfc_accuracies)

# %%
print("Mean Accuracies", rfc_accuracies.mean()*100)

# %%
rfc_pred = rfc.predict(X_test)
rfc_pred

# %%
# confusion matrix
cm = metrics.confusion_matrix(y_test, rfc_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index=[i for i in [0, 1]], columns=[i for i in ['Predicted 0', 'Predicted 1']])
plt.figure(figsize=(7, 5))
sns.heatmap(df_cm, annot=True, cmap='coolwarm',fmt='.2f', linecolor='k', linewidths=0.2)

# %% [markdown]
# ### **KNeighbors Classifier**

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GridSearchCV

param_grid = {'leaf_size' : [2, 5, 7, 9, 11], 'n_neighbors' : list(range(1, 11)), 'p' : [1, 2]}
gs = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid)
gs.fit(X_train, y_train)

# %%
# get the best parameters after the hyperparameter tuning
gs.best_params_

# %%
# build Neighbors algorithm
Knn = KNeighborsClassifier(n_neighbors=1, p=1, leaf_size=2)
Knn.fit(X_train, y_train)

# %%
knn_accuracies = cross_val_score(estimator=Knn, X=X_train, y=y_train, cv=10)
print("Accuracies:\n", knn_accuracies)

# %%
print("Mean Accuracies:", knn_accuracies.mean()*100)

# %%
Knn_pred = Knn.predict(X_test)
Knn_pred

# %%
# Confusion matrix

cm = metrics.confusion_matrix(y_test, Knn_pred, labels=[0, 1])
df_cm = pd.DataFrame(cm, index=[i for i in [0, 1]], columns=[i for i in ['Predicted 0', 'Predicted 1']])
plt.figure(figsize=(7, 5))
sns.heatmap(df_cm, annot=True, cmap='coolwarm',fmt='.2f', linecolor='k', linewidths=0.2)

# %% [markdown]
# ## **Printing each algorithm and the accuracy score**

# %%
print("LogisticRegression : {0:.2f}%".format(accuracies.mean()*100))
print("SupportVectorMachine_1(Kernel='liblinear') : {0:.2f}%".format(svm_accuracies.mean()*100))
print("SupportVectorMachine_2(Kernel='rbf'): {0:.2f}%".format(rbf_accuracies.mean()*100))
print("RandomForestClassifier: {0:.2f}%".format(rfc_accuracies.mean()*100))
print("KNeighborsClassifier: {0:.2f}%".format(knn_accuracies.mean()*100))


