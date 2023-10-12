import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

train = pd.read_csv('layer_10_train.csv')
valid = pd.read_csv('layer_10_valid.csv')
test = pd.read_csv('layer_10_test.csv')

# preprocessing
columns_to_check = ['label_3']
train = train.dropna(subset=columns_to_check, how='any')

datasets = {'train': train, 'valid': valid, 'test': test}

for name, dataset in datasets.items():
    datasets[name] = dataset.fillna(dataset.mean())


# separate features and labels

def separate_features_labels(data):
    features = data.drop(columns=['label_1', 'label_2', 'label_3', 'label_4'])
    labels = data[['label_3']]
    return features, labels


# id_column = test[['ID']]
label3_train_X, label3_train_y = separate_features_labels(train)
label3_valid_X, label3_valid_y = separate_features_labels(valid)
test_X = test.drop(columns=['label_1', 'label_2', 'label_3', 'label_4'])

# initial accuracy check
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(label3_train_X)
valid_X_scaled = scaler.transform(label3_valid_X)
test_X_scaled = scaler.transform(test_X)

print(train_X_scaled.shape)

model_initial = SVC()
model_initial.fit(train_X_scaled, label3_train_y)

y_pred = model_initial.predict(valid_X_scaled)

accuracy_intial = accuracy_score(label3_valid_y, y_pred)

y_pred_test_label3 = model_initial.predict(test_X_scaled)

print("initial Accuracy:", accuracy_intial)

# feature extraction

# Calculate the correlation matrix of features

correlation_matrix = label3_train_X.corr()

correlation_threshold = 0.5
# Create a boolean mask indicating highly correlated features
mask = np.abs(correlation_matrix) > correlation_threshold

# Exclude the diagonal and upper triangular part to avoid redundancy
mask = np.triu(mask, k=1)

# Find column names of highly correlated features
highly_correlated = set(correlation_matrix.columns[mask.any(axis=0)])

# removing high correlated features

label3_train_X = label3_train_X.drop(columns=highly_correlated)
label3_valid_X = label3_valid_X.drop(columns=highly_correlated)
test_X = test_X.drop(columns=highly_correlated)

# standardize

scaler = StandardScaler()

label3_train_features_standardized = scaler.fit_transform(label3_train_X)
label3_valid_features_standardized = scaler.transform(label3_valid_X)
label3_test_features_standardized = scaler.transform(test_X)

print(label3_train_X.shape)

# Accuracy after feature extraction
model_next = SVC()
model_next.fit(label3_train_features_standardized, label3_train_y)

y_pred = model_next.predict(label3_valid_features_standardized)

accuracy = accuracy_score(label3_valid_y, y_pred)

print("Accuracy after feature extraction:", accuracy)


# define method to create the dataframe and save it as a csv file
def create_csv(pred_after_fe):
    df = pd.DataFrame()

    # df.insert(loc=0, column='ID', value=id_column)
    df.insert(loc=0, column='label_3', value=pred_after_fe)

    df.to_csv('output/output_label_3.csv', index=False)
    print("Saved successfully to CSV")


create_csv(y_pred_test_label3)
