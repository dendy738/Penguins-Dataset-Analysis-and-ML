import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, fbeta_score, confusion_matrix, get_scorer_names
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

score_names = get_scorer_names()
print(score_names)

# ====== Load DataSet ========
penguins = sns.load_dataset("penguins")

print(penguins.head())

# ======= EDA ===========
def data_eda(data):
    print('-' * 10, 'Data Set Information', '-' * 10)
    data.info()
    print('\n', '-' * 10, 'Data Set Describe', '-' * 10)
    print(data.describe())
    print()
    print(data.describe(include='object'))
    print('\n', '-' * 10, 'Missing Values', '-' * 10)
    print(data.isnull().sum())

data_eda(penguins)

# ========= Visual ==========
sns.set_style('darkgrid')
sns.set_palette('colorblind')
sns.set_context('notebook')
sns.set_palette(sns.color_palette("hls", 8))

def distribution(data):
    num_cols = [x for x in data.select_dtypes(include='number').columns]

    for sp in data['species'].unique():
        fig = plt.figure(figsize=(10, 6), facecolor='skyblue')
        for col in range(len(num_cols)):
            plt.subplot(2, 2, col + 1)
            sns.histplot(data[data['species'] == sp], x=num_cols[col])
            plt.title(f'Distribution of {num_cols[col]}')

        fig.suptitle(f'Distribution of parameters of {sp}')
        plt.tight_layout()
        plt.show()

distribution(penguins)

def box_plot(data):
    plt.figure(figsize=(9, 7), facecolor='skyblue')

    idx = 1
    for col in data.select_dtypes(include='number').columns:
        plt.subplot(2, 2, idx)
        sns.boxplot(data, x='species', y=col, width=0.6)
        plt.title(f'Distribution of {col} by Species')
        idx += 1

    plt.tight_layout()
    plt.show()

box_plot(penguins)

'''
A few outliers are present in each feature. So, we will scale of values.
'''

# ========= Drop Missing Values ==========

penguins.dropna(subset=[x for x in penguins.select_dtypes(include='number').columns], inplace=True)

'''
We delete only those instances that have many missing values.
Other missing values will imputed by Imputer objects in pipelines.
'''

# =========== Feature Engineering ==========

penguins['bill_length_cm'] = penguins['bill_length_mm'] / 10
penguins['bill_depth_cm'] = penguins['bill_depth_mm'] / 10
penguins['flipper_length_cm'] = penguins['flipper_length_mm'] / 10
penguins['body_mass_kg'] = penguins['body_mass_g'] / 1000
penguins.drop(columns=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'], inplace=True)

# ========== Split dataset ===========

features = penguins.drop(['species'], axis=1)
target = penguins['species']
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3, random_state=39)


# ========== Create Pipelines =============

numeric_features = [x for x in features.select_dtypes(include='number')]
cat_features = [x for x in features.select_dtypes(include='object')]
num_pipeline = Pipeline([('imputer', KNNImputer(n_neighbors=3)), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(sparse_output=False, drop='first'))])
preprocessor = ColumnTransformer(transformers=[('numeric', num_pipeline, numeric_features), ('category', cat_pipeline, cat_features)])

# ============ Initialize Models =============

model_1 = RandomForestClassifier(max_depth=3, random_state=39)
model_2 = KNeighborsClassifier(n_neighbors=5)
kf = KFold(n_splits=4, shuffle=True, random_state=42)

# MODEL 1: USE CROSS-VALIDATION
full_pipe = Pipeline([('preproc', preprocessor), ('classifier', model_1)])
full_pipe.fit(features_train, target_train)
print(full_pipe.__sklearn_is_fitted__())
score = cross_val_score(full_pipe, features_train, target_train, scoring='accuracy', cv=kf)
pred = full_pipe.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print(f'Scores of all iterations: {score}')
print(f'Mean value of all scores: {np.mean(score)}')
print(f'Model accuracy: {accuracy}')



# MODEL 2: USE USUAL TRAINGIN SET
full_pipe_2 = Pipeline([('preproc', preprocessor), ('classifier', model_2)])
full_pipe_2.fit(features_train, target_train)
pred_1 = full_pipe_2.predict(features_test)
accuracy_2 = accuracy_score(target_test, pred_1)
score2 = full_pipe_2.score(features_test, target_test)
print(f'Accuracy of model 2: {accuracy_2} ')
print(f'Score of model 2: {score2} ')


# ========== Models test ===========

# CREATE RANDOM SAMPLES
random_data = penguins.sample(150, random_state=150)
extra_features = random_data.drop(['species'], axis=1)
extra_target = random_data['species']

# CHECK ACCURACY ON THE RANDOM TEST SAMPLES(MODEL 1 AND MODEL 2)
extra_pred_model_1 = full_pipe.predict(extra_features)
extra_pred_model_2 = full_pipe_2.predict(extra_features)
acc1 = accuracy_score(extra_target, extra_pred_model_1)
acc2 = accuracy_score(extra_target, extra_pred_model_2)
print(f'Accuracy of model 1: {acc1}')
print(f'Accuracy of model 2: {acc2}')

# =========== GridSearchCV ============
# For model 1
grid_param = {
    'preproc__numeric__imputer__n_neighbors': [2, 3, 4, 5, 6],
    'classifier__n_estimators': [2, 3, 5, 10, 20, 50, 100, 200],
    'classifier__max_depth': [2, 3, 5, 10, None]
}
grid_s = GridSearchCV(full_pipe, grid_param, cv=kf, verbose=1, scoring='accuracy')
grid_s.fit(features, target)

# For model 2
grid_params_2 = {
    'preproc__numeric__imputer__n_neighbors': [2, 3, 4, 5, 6],
    'classifier__n_neighbors': [1, 2, 3, 4, 5, 7, 10],
    'classifier__p': [1, 2, 3, 5]
}
grid_s_1 = GridSearchCV(full_pipe_2, grid_params_2, scoring='accuracy', cv=kf, verbose=1)
grid_s_1.fit(features, target)

print(f'Best parameters(model 1): {grid_s.best_params_}')
print(f'Best score (model 1): {grid_s.best_score_}')
print(f'Best parameters(model 2): {grid_s_1.best_params_}')
print(f'Best score (model 2): {grid_s_1.best_score_}')

best_model_1 = grid_s.best_estimator_
best_model_2 = grid_s_1.best_estimator_

# ========== Evaluate Models ============
add_random_samples = penguins.sample(200, random_state=156)
F, T = add_random_samples.drop(['species'], axis=1), add_random_samples['species']

model_1_pred = best_model_1.predict(F)
model_2_pred = best_model_2.predict(F)

# Precision
prec_1 = precision_score(T, model_1_pred, average='macro')
prec_2 = precision_score(T, model_2_pred, average='macro')
print(f'Precision of model 1: {prec_1}')
print(f'Precision of model 2: {prec_2}')

# Recall
recall_1 = recall_score(T, model_1_pred, average='macro')
recall_2 = recall_score(T, model_2_pred, average='macro')
print(f'Recall of model 1: {recall_1}')
print(f'Recall of model 2: {recall_2}')

# F1 score
f1_model_1 = f1_score(T, model_1_pred, average='macro')
f1_model_2 = f1_score(T, model_2_pred, average='macro')
print(f'F1 score of model 1: {f1_model_1}')
print(f'F1 score of model 2: {f1_model_2}')

# Confusion matrix
conf_matrix_model_1 = confusion_matrix(T, model_1_pred)
conf_matrix_model_2 = confusion_matrix(T, model_2_pred)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_model_1, annot=True)
plt.title('Confusion matrix of model 1')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_model_2, annot=True)
plt.title('Confusion matrix of model 2')
plt.ylabel('Actual')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()










