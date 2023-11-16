# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import hopsworks
import pandas as pd
# %matplotlib inline

# %%
project = hopsworks.login()
fs = project.get_feature_store()

# %%
wine_df = pd.read_csv("data/wine.csv")

# %%
wine_df.info()

# %% [markdown]
# **From this we can gather two important issues that need to be handled**
# 1. The feature `type` is of type object and needs to be handled differently  
# than the other features  
# 2. There are a couple of nan values for some of the feature categories these  
#  need to be handled in som type of way.
#
#
#
# Lets handle the first issue by introducing a dummy variable i.e. introduce  
# one feature that is type-white and another that is type-red and let them be  
# 0 or 1. This is often called one-hot encoding.

# %%
dummy_variables_type = pd.get_dummies(wine_df.type,prefix='type',dtype='int')
dummy_variables_type

# %%
wine_df = pd.concat([dummy_variables_type,wine_df],axis=1)
wine_df = wine_df.drop('type',axis=1)

# %%
wine_df

# %% [markdown]
# Lets now deal with the features where we have nan values

# %%
for col in wine_df.columns:
    
    n_null_values = wine_df[col].isnull().sum()
    
    if n_null_values > 0:
        print(col, n_null_values)

# %% [markdown]
# Since there are only a few nan values over the entire dataset it is sufficient to  
# just replace those nan values with the average value for the same feature and quality.

# %%
mean_feature_value_per_qualityGroup = wine_df.groupby('quality').mean()
mean_feature_value_per_qualityGroup

# %%
for col in wine_df.columns:

    nan_indexes = wine_df[wine_df[col].isnull()]

    for index, row in nan_indexes.iterrows():

        quality = row.quality
        mean_value = mean_feature_value_per_qualityGroup.at[quality, col]

        wine_df.at[index, col] = mean_value

# %%
wine_df.isna().sum()

# %% [markdown]
# Now there are no more nan values.
#
# Lets also check for duplicates and remove them if they do exist. 

# %%
wine_df.duplicated().sum()

# %%
wine_df.duplicated().sum()/len(wine_df)

# %% [markdown]
# We apparantly have quite a substantial number of duplicates almost 18 %, lets remove them

# %%
wine_df = wine_df.drop_duplicates()

# %%
wine_df.duplicated().sum()

# %% [markdown]
# We have now dropped 1168 out of 6497 values but at least we have no more duplicates.
# We can now consider the data cleaned enough for an EDA 
#
# Lets take a quick look at some of the simple statistics of the dataset.

# %%
wine_df.describe().transpose()

# %%
wine_df['quality'].value_counts().sort_index()

# %% [markdown]
# ### Exploratory Data Analysis (EDA) on the wine data
#
# Let's look at our wines - the distribution and range of values for the 13 different features
#  * type_red
#  * type_white
#  * fixed acidity
#  * volatile acidity
#  * citric acid
#  * residual sugar
#  * chlorides
#  * free sulfur dioxide
#  * total sulfur dioxide
#  * density
#  * pH
#  * sulphates
#  * alcohol
#  
#  and the target variable is `quality`.

# %% [markdown]
# #### First lets have a look at the target variable

# %%
sns.histplot(wine_df, x='quality')

# %% [markdown]
# We can tell that there are very few values in the categories 3 and 9.  
#
# We can also tell that the majority of the values are in classes 5 or 6.  
#
# It is important to keep this uneven distribution in mind for when we  
# decide to create a model. A naive classifier that always responds 6  
# would do "ok" on this dataset, however would probably be horrible   
# in comparison to real world data. 

# %% [markdown]
# ##### Then lets have a look at the correlation matrix

# %%
plt.figure(figsize=(9,6))
sns.heatmap(wine_df.corr(),annot=True,linewidths=0.5,cmap='coolwarm')

# %% [markdown]
# While there is some correlations we have no reason to suspect multi-colinearity  
# for any of the variables except two. Type_red and Type_white have a -1 correlation  
# and therefore one of them should be removed for the model.

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
wdf2 = wine_df2
#wdf2 = wdf2.drop('quality',axis=1)
wdf2 = wdf2[['total sulfur dioxide','free sulfur dioxide','quality']].drop('free sulfur dioxide',axis=1)
# wdf2 = wdf2.drop('type_red',axis=1)
# wdf2 = wdf2.drop('density',axis=1)
# wdf2 = wdf2.drop('pH',axis=1)

# %%
vif_data = pd.DataFrame() 
vif_data["feature"] = wdf2.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

vif_data

# %% [markdown]
# This can help us reduce colinearity within the dataset

# %% [markdown]
# Next step is to look into the correlations with the target value specifically

# %%
plt.figure(figsize=(10,4))

# We drop quality on the row axis and then uses the correlations on the column axis
correlation_matrix = wine_df.corr().drop('quality')['quality']
g = sns.barplot(correlation_matrix)

# Rotate the coloumn names so that we can see them in the plot.
g.set_xticklabels(g.get_xticklabels(),rotation=45)
plt.show()

# %% [markdown]
# #### Correlation "significance"

# %%
feature_cols = wine_df.drop('quality',axis=1).columns

corrs_and_pvalues = [scipy.stats.pearsonr(wine_df['quality'],wine_df[feature_col]) for feature_col in feature_cols]
corrs, pvalues = zip(*corrs_and_pvalues)

significance_df = pd.DataFrame()
significance_df['feature'] = feature_cols
significance_df['correlation with quality'] = corrs
significance_df['p value of correlation'] = pvalues
significance_df

# %%
From what we can see there are a number of the correlations that are not significant  
We can remove these from the featurelist as they will probably not help the model.

# %%
significance_df['p > 0.05'] = [p > 0.05 for p in pvalues]
significance_df['p > 0.01'] = [p > 0.01 for p in pvalues]
significance_df['p > 0.001'] = [p > 0.001 for p in pvalues]
significance_df[significance_df['p > 0.05']]

# %%
significance_df[significance_df['p > 0.01']]

# %%
significance_df[significance_df['p > 0.001']]

# %% [markdown]
# #### Based on the above we can notice the following
# The features `sulphates` and `ph` have a low correlation with quality and  
# this correlation is not considered significant at the 99,9 % confidence level.  
# It `could be removed`, but this is probably to be decided on the validation set.
#
# However, it is not obvious what should be removed.

# %% [markdown]
# Here we can pairplot them as well, however, it seems quite unneccessary since it is  
# difficult to keep track of what happens.  
#
# If there are any interesting relationships we want to look closer at, we can slice the  
# dataframe and chose to keep only the relevant features
#

# %%
features_to_look_into = ['alcohol','volatile acidity'] + ['quality']

g = sns.PairGrid(wine_df[features_to_look_into], hue='quality', palette='GnBu_d',diag_sharey=False, corner=True)
g.map_lower(sns.scatterplot)
g.map_diag(sns.kdeplot,multiple="stack")
g.add_legend()
