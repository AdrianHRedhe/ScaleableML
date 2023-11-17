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
from statsmodels.stats.outliers_influence import variance_inflation_factor
# %matplotlib inline

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
# The first and most important conclusion from this correlation matrix is that  
# there is a 1:1 correlation between type_white and type_red, this is to be expected  
# but since there are only two types of wine, it makes sense to remove type_red to  
# avoid colinearity.

# %% [markdown]
# #### Removing further colinearity
# We can use variance inflation factor to further explore the colinearity of our data  
# a high value of *VIF* leads signifies that the "independent" variables are highly 
# related and this can in turn lead to worse prediction power in the models.

# %%
wdf2 = wine_df.copy()
wdf2 = wdf2.drop('quality',axis=1)
# wdf2 = wdf2[['total sulfur dioxide','free sulfur dioxide']]#.drop('free sulfur dioxide',axis=1)
wdf2 = wdf2.drop('type_red',axis=1)
wdf2 = wdf2.drop('density',axis=1)
wdf2 = wdf2.drop('pH',axis=1)

# %%
vif_data = pd.DataFrame() 
vif_data["feature"] = wdf2.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(wdf2.values, i) for i in range(len(wdf2.columns))]

vif_data.sort_values('VIF',ascending=False)

# %%
# plt.figure(figsize=(9,6))
# sns.heatmap(wine_df.drop(['quality'],axis=1).corr(),annot=True,linewidths=0.5,cmap='coolwarm')

# %%
# plt.figure(figsize=(9,6))
# sns.heatmap(wdf2.corr(),annot=True,linewidths=0.5,cmap='coolwarm')

# %% [markdown]
# ### Conclusions from looking at the colinearity
# After looking at the *VIF* values of this dataset, the conclusion is  
# that the dataset is very very colinear. It makes sense given that many  
# of the categories are related.
#
# Take `fixed acidity`, `volatile acidity`, `citric acid` and `pH`  
# for example they are all related to acidity, and we can expect that if the  
# `fixed acidity` goes up then the `pH` value will go down.
#
# It goes somewhat beyond the scope of this assignment to further break  
# down this analysis, but I think spending time on this in particular can be  
# very helpful for model generation in the future.
#
# The conclusions on a level that is reasonable to this report is that `pH`  
# is highly colinear and `density` as well. These two features could be  
# removed, lets look at the rest of the EDA first.

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
significance_df.sort_values('p value of correlation')

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
# They could be removed, especially `pH` that was shown to be highly colinear  
# as well

# %% [markdown]
# Here we can pairplot them as well, however, it seems quite unneccessary since it is  
# difficult to keep track of what happens.  
#
# If there are any interesting relationships we want to look closer at, we can slice the  
# dataframe and chose to keep only the relevant features
#

# %%
features_to_look_into = ['alcohol','volatile acidity','chlorides'] + ['quality']

# g = sns.PairGrid(wine_df, hue='quality', palette='GnBu_d',diag_sharey=False, corner=True)
g = sns.PairGrid(wine_df[features_to_look_into], hue='quality', palette='GnBu_d',diag_sharey=False, corner=True)
g.map_lower(sns.scatterplot,marker='.')
g.map_diag(sns.kdeplot,multiple="stack")
g.add_legend()

# %% [markdown]
# ## Conclusions on the EDA
#
# Features to be removed: type_red
#
# Features that could be removed: density,pH based on VIF & pH, sulphates based on pearson corr p value
# Decisions on the features above:
# * **Keep** *Density*, has very high VIF but atleast high correlation and low p-value
# * **Remove** *pH* based on very high VIF, low correlation and high p-value
# * **Remove** *sulphates* based on medium high VIF, low correlation and high p-value
#
#
# Extra feature engineering and EDA should spend time on this:
# * Handling the skewed target value group
# * Spending time on breaking up Collinearity, by dropping or changing some features
# * Investigating if data should be scaled and reweighted

# %%
wine_df = wine_df.drop(['type_red','pH','sulphates'],axis=1)

# %% [markdown]
# ### Insert our Wine DataFrame into a FeatureGroup
# Let's write our historical wine feature values and labels to a feature group.
# When you write historical data, this process is called `backfilling`.

# %%
project = hopsworks.login()
fs = project.get_feature_store()

# %%
# Rename columns in order to fit in with hopsworks conventions
cols_no_spaces = [col.replace(' ','_') for col in wine_df.columns]
wine_df.columns = cols_no_spaces
feature_cols_no_spaces = wine_df.drop('quality',axis=1).columns

# %%
wine_fg = fs.get_or_create_feature_group(
    name="wine",
    version=1,
    primary_key=feature_cols_no_spaces,
    description="Wine quality dataset")

wine_fg.insert(wine_df)
