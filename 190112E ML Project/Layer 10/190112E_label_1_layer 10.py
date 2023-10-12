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
columns_to_check = ['label_1']
train = train.dropna(subset=columns_to_check, how='any')

datasets = {'train': train, 'valid': valid, 'test': test}

for name, dataset in datasets.items():
    datasets[name] = dataset.fillna(dataset.mean())


# separate features and labels

def separate_features_labels(data):
    features = data.drop(columns=['label_1', 'label_2', 'label_3', 'label_4'])
    labels = data[['label_1']]
    return features, labels


# id_column = test[['ID']]
label1_train_X, label1_train_y = separate_features_labels(train)
label1_valid_X, label1_valid_y = separate_features_labels(valid)
test_X = test.drop(columns=['label_1', 'label_2', 'label_3', 'label_4'])

# initial accuracy check
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(label1_train_X)
valid_X_scaled = scaler.transform(label1_valid_X)

print(train_X_scaled.shape)

model_initial = SVC()
model_initial.fit(train_X_scaled, label1_train_y)

y_pred = model_initial.predict(valid_X_scaled)

accuracy_intial = accuracy_score(label1_valid_y, y_pred)

print("initial Accuracy:", accuracy_intial)

# feature extraction

# Calculate the correlation matrix of features

correlation_matrix = label1_train_X.corr()

correlation_threshold = 0.5
# Create a boolean mask indicating highly correlated features
mask = np.abs(correlation_matrix) > correlation_threshold

# Exclude the diagonal and upper triangular part to avoid redundancy
mask = np.triu(mask, k=1)

# Find column names of highly correlated features
highly_correlated = set(correlation_matrix.columns[mask.any(axis=0)])

# removing high correlated features

label1_train_X = label1_train_X.drop(columns=highly_correlated)
label1_valid_X = label1_valid_X.drop(columns=highly_correlated)
test_X = test_X.drop(columns=highly_correlated)

# standardize

scaler = StandardScaler()

label1_train_features_standardized = scaler.fit_transform(label1_train_X)
label1_valid_features_standardized = scaler.transform(label1_valid_X)
label1_test_features_standardized = scaler.transform(test_X)

print(label1_train_X.shape)

# Accuracy after feature extraction
model_next = SVC()
model_next.fit(label1_train_features_standardized, label1_train_y)

y_pred = model_next.predict(label1_valid_features_standardized)

accuracy = accuracy_score(label1_valid_y, y_pred)

print("after drop corr Accuracy:", accuracy)

# PCA

# Set the variance threshold for PCA
variance_threshold = 0.95

# Create a PCA transformer with the specified variance threshold
pca_transformer = PCA(n_components=variance_threshold, svd_solver='full')

# Apply PCA transformation to standardized features
label1_pca_train_features_transformed = pca_transformer.fit_transform(label1_train_features_standardized)
label1_pca_valid_features_transformed = pca_transformer.transform(label1_valid_features_standardized)
label1_pca_test_features_transformed = pca_transformer.transform(label1_test_features_standardized)

print(label1_pca_train_features_transformed.shape)

# hyper-parameter tuning

# Define the hyperparameters and their possible value distributions
param_dist = {
    'C': np.logspace(-3, 3, 1000),  # Regularization parameter (log-scale)
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Kernel type
    'gamma': np.logspace(-3, 3, 1000)  # Gamma parameter (log-scale)
}

# Create an SVM model
svm = SVC()

# Convert DataFrame to 1-dimensional arrays (Series)
label1_train_y = label1_train_y.values.ravel()
label1_valid_y = label1_valid_y.values.ravel()

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', verbose=2,
                                   n_jobs=-1, random_state=42)

# Perform random search on the training data
random_search.fit(label1_pca_train_features_transformed, label1_train_y)

# Get the best hyperparameters from the random search
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Train the best model on the entire training dataset
best_model.fit(label1_pca_train_features_transformed, label1_train_y)

# Predict on the train data using the best model
y_pred_train_label1 = best_model.predict(label1_pca_train_features_transformed)

# Calculate accuracy on the train data
accuracy_train = accuracy_score(label1_train_y, y_pred_train_label1)

# Predict on the validation data using the best model
y_pred_valid_label1 = best_model.predict(label1_pca_valid_features_transformed)

# Calculate accuracy on the validation data
accuracy_valid = accuracy_score(label1_valid_y, y_pred_valid_label1)

y_pred_test_label1 = best_model.predict(label1_pca_test_features_transformed)

print("Accuracy on train data:", accuracy_train)
print("Accuracy on validation data:", accuracy_valid)

# Create a DataFrame from the best_params dictionary
best_params_df = pd.DataFrame([best_params])

# Save the DataFrame to a CSV file
best_params_df.to_csv('best_params1.csv', index=False)

print("Best params are: ", best_params)
print("Best model is: ", best_model)

# Best model
# best_params = {'kernel': 'poly', 'gamma': 46.41588833612782, 'C': 2.124845352498883}
# best_model = SVC(**best_params)
#
# # Train the best model on the entire training dataset
# best_model.fit(label1_pca_train_features_transformed, label1_train_y)
#
# # Predict on the train data using the best model
# y_pred_train_label1 = best_model.predict(label1_pca_train_features_transformed)
#
# # Calculate accuracy on the train data
# accuracy_train = accuracy_score(label1_train_y, y_pred_train_label1)
#
# # Predict on the validation data using the best model
# y_pred_valid_label1 = best_model.predict(label1_pca_valid_features_transformed)
#
# # Calculate accuracy on the validation data
# accuracy_valid = accuracy_score(label1_valid_y, y_pred_valid_label1)
#
# y_pred_test_label1 = best_model.predict(label1_pca_test_features_transformed)
#
# print("Accuracy on train data:", accuracy_train)
# print("Accuracy on validation data:", accuracy_valid)


# define method to create the dataframe and save it as a csv file
def create_csv(pred_after_fe):
    df = pd.DataFrame()

    # df.insert(loc=0, column='ID', value=id_column)
    df.insert(loc=0, column='label_1', value=pred_after_fe)

    df.to_csv('output/output_label_1.csv', index=False)
    print("Saved successfully to CSV")


create_csv(y_pred_test_label1)
