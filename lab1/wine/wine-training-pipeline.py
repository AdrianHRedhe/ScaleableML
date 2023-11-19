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
import hopsworks
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from matplotlib import pyplot
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib
import os

# %% [markdown]
# ### In this notebook we are going to do 3 things:
# 1. Create a feature view
# 2. Train several classifiers and regression models and evaluate their performance on the test set
# 3. Upload one of the better models to Hopsworks to be used with the other programs.

# %%
# Login to hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# %%
# The feature view is the input set of features for your model. The features can come from different feature groups.    
# You can select features from different feature groups and join them together to create a feature view

wine_fg = fs.get_feature_group(name='wine', version=1)
query = wine_fg.select_all()
feature_view = fs.get_or_create_feature_view(name='wine',
                                  version=1,
                                  description='Read from Wine quality dataset',
                                  labels=['quality'],
                                  query=query)

# %% [markdown]
# ### Splitting the dataset and trying out our models

# %%
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

# %%
# Lets define our models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

clf1 = DecisionTreeClassifier(max_depth=5)
p1 = {'max_depth': [1, 5, 10, 20]}
clf2 = KNeighborsClassifier(n_neighbors=4)
p2 = {'n_neighbors': [1, 3, 5, 7]}
clf3 = RandomForestClassifier()
p3 = {'max_depth': [1, 5, 10, 20]}
clf4 = SVC()
p4 = {'kernel': ['linear','rbf']}

models = [clf1,clf2,clf3,clf4]
params_to_be_tuned = [p1,p2,p3,p4]


# %%
def hyperparameter_tuning(model, params, X_train, y_train):
    gs = GridSearchCV(model,
                        param_grid=params,
                        scoring='accuracy',
                        cv=3)

    gs.fit(X_train,y_train)
    return gs


# %% [markdown]
# ### Training and tuning the models

# %%
tuned_models = []
for i in range(len(models)):
    print(f'Tuning model {i+1}')
    tuned_model = hyperparameter_tuning(models[i], params_to_be_tuned[i], X_train, y_train.values.ravel())
    tuned_models.append(tuned_model)

# %% [markdown]
# ### Evaluating the models

# %%
preds = [model.predict(X_test) for model in tuned_models]
accs = [accuracy_score(y_test,y_pred) for y_pred in preds]
f1s = [f1_score(y_test,y_pred,average='weighted') for y_pred in preds]

print(accs)
print(f1s)

# %%
n_max_class = y_test.reset_index().groupby('quality').count()['index'].max()
n_total = len(y_test)
benchmark = n_max_class/n_total

print(benchmark)

# %% [markdown]
# We can see that the models do perform better than the benchmark, the benchmark here being a very naive  
# approach where we just classify everything as the most common class. 
#
# The benchmark is **45.3 %** and the best performing model did get **57.5 %** accuracy on the test set.
#
# While this is not a lot better than the benchmark, it does seem like the bulk of the extra work should  
# actually go to feature engineering, like cutting down or changing features and reducing colinearity,   
# and not to tune parameters and selecting models.
#
# Another thing that would be of a very high level of importance would be to reframe the problem, possibly  
# as a binary classification of good wines vs bad wines where the classes are decided on a certain number  
# of the quality label.
#
# Below we can see the model that performed the best and the hyperparameter that worked best for this task.

# %%
best_model = tuned_models[2]
print(best_model.estimator, best_model.best_params_)

# %% [markdown]
# Lets also take a more in depth look at the metrics for each of the classes  
# and also a look at a confusion matrix

# %%
y_pred = best_model.predict(X_test)

metrics = classification_report(y_test, y_pred, output_dict=True, zero_division=True)

pd.DataFrame(metrics).transpose()

# %%
rows = [f'True_{classNumber}' for classNumber in range(3,10)]
cols = [f'Pred_{classNumber}' for classNumber in range(3,10)]

conf_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix_df = pd.DataFrame(conf_matrix,rows,cols)

pyplot.figure(figsize=(10,5))
g = sns.heatmap(confusion_matrix_df,annot=True,lw=.5)
fig = g.get_figure()

# %% [markdown]
# As previously hypothesised we can see that the recall for any other class than 5 and 6 is abysmally low.
# We would probably need more data, or to reframe the question somewhat in order for models to do much better  
#
# However, it is still ok performance given the scope of the assignment.
#
# Lets **save** this model to a folder in the repository and lets also **upload it to hopsworks**

# %%
# We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
mr = project.get_model_registry()

# The contents of the 'wine_model' directory will be saved to the model registry. Create the dir, first.
model_dir="wine_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

# Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
joblib.dump(best_model, model_dir + "/wine_model.pkl")
fig.savefig(model_dir + "/confusion_matrix.png")    

# Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

# Create an entry in the model registry that includes the model's name, desc, metrics
wine_model = mr.python.create_model(
    name="wine_model", 
    metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    description="Wine Quality Predictor"
)

# Upload the model to the model registry, including all files in 'model_dir'
wine_model.save(model_dir)
