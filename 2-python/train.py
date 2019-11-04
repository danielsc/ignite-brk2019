# ### Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn_pandas import DataFrameMapper
import os
import pandas as pd

# +
from azureml.core import Run, Workspace, Experiment, Dataset
# Check core SDK version number
import azureml.core

print("SDK version:", azureml.core.VERSION)
# -

OUTPUT_DIR='./outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# +
print('load dataset')

run = Run.get_context()
if(run.id.startswith("OfflineRun")):
    ws = Workspace.from_config()
    experiment = Experiment(ws, "Train-Interactive")
    is_remote_run = False
    run = experiment.start_logging(outputs=None, snapshot_directory=".")
    ds = ws.datasets['IBM-Employee-Attrition']
else:
    ws = run.experiment.workspace
    ds = run.input_datasets['attrition']
    is_remote_run = True

attritionData = ds.to_pandas_dataframe()
print(attritionData.head())
# -

# Dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# Dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)

attritionData = attritionData.drop(['Over18'], axis=1)

# Since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)
target = attritionData["Attrition"]

attritionXData = attritionData.drop(['Attrition'], axis=1)

# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attritionXData.columns.difference(categorical)

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in numerical]

categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in categorical]

transformations = numeric_transformations + categorical_transformations

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', LogisticRegression(solver='lbfgs'))])


# Split data into train and test
x_train, x_test, y_train, y_test = train_test_split(attritionXData,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)


print('train model')
# preprocess the data and fit the classification model
clf.fit(x_train, y_train)

run.log("accuracy", clf.score(x_test, y_test))

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# -

cm = confusion_matrix(y_test, clf.predict(x_test))
print(cm)
fig = print_confusion_matrix(cm, class_names=["not leaving","leaving"])
run.log_image("confusion", plot=fig)

# +
# save model for use outside the script
model_file_name = 'log_reg.pkl'
joblib.dump(value=clf, filename=model_file_name)

# register the model with the model management service for later use
run.upload_file(model_file_name, model_file_name)
# -

input_dataset = ds.take(100).drop_columns('Attrition')
output_dataset = ds.take(100).keep_columns('Attrition')

# +
from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

azureml_model = run.register_model(model_name='attrition_python',
                                   model_path=model_file_name,
                                   description='Logistic regression Scikit-Learn model for Attrition prediction.',
                                   tags={'area': 'attrition', 'type': 'classification'},
                                   datasets=[(Dataset.Scenario.TRAINING,ds)])

print('Name:', azureml_model.name)
print('Version:', azureml_model.version)
# # +


if not is_remote_run:
    run.complete()

print('completed')

# -


