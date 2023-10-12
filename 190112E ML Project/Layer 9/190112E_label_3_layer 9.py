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

train = pd.read_csv('layer_9_train.csv')
valid = pd.read_csv('layer_9_valid.csv')
test = pd.read_csv('layer_9_test.csv')

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

# test_X = test.drop(columns=['ID'])

# initial accuracy check
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(label3_train_X)
valid_X_scaled = scaler.transform(label3_valid_X)
test_X_scaled = scaler.transform(test_X)

print(train_X_scaled.shape)

model_initial = SVC()
model_initial.fit(train_X_scaled, label3_train_y.values.ravel())

y_pred = model_initial.predict(valid_X_scaled)

accuracy_intial = accuracy_score(label3_valid_y, y_pred)

y_pred_test_label3 = model_initial.predict(test_X_scaled)

print("Accuracy initial:", accuracy_intial)

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
label3_train_y = label3_train_y.values.ravel()
label3_valid_y = label3_valid_y.values.ravel()

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(svm, param_distributions=param_dist, n_iter=5, cv=5, scoring='accuracy', verbose=2,
                                   n_jobs=-1, random_state=42)

# Perform random search on the training data
random_search.fit(label3_train_X, label3_train_y)

# Get the best hyperparameters from the random search
best_params = random_search.best_params_
best_model = random_search.best_estimator_

# Train the best model on the entire training dataset
best_model.fit(label3_train_X, label3_train_y)

# Predict on the train data using the best model
y_pred_train_label3 = best_model.predict(label3_train_X)

# Calculate accuracy on the train data
accuracy_train = accuracy_score(label3_train_y, y_pred_train_label3)

# Predict on the validation data using the best model
y_pred_valid_label3 = best_model.predict(label3_valid_X)

# Calculate accuracy on the validation data
accuracy_valid = accuracy_score(label3_valid_y, y_pred_valid_label3)

y_pred_test_label3 = best_model.predict(test_X)

print("Accuracy on train data:", accuracy_train)
print("Accuracy on validation data:", accuracy_valid)

# Create a DataFrame from the best_params dictionary
best_params_df = pd.DataFrame([best_params])

# Save the DataFrame to a CSV file
best_params_df.to_csv('best_params3.csv', index=False)

print("Best params are: ", best_params)
print("Best model is: ", best_model)


# define method to create the dataframe and save it as a csv file
def create_csv(pred_after_fe):
    df = pd.DataFrame()

    # df.insert(loc=0, column='ID', value=id_column)
    df.insert(loc=0, column='label_3', value=pred_after_fe)

    df.to_csv('output/output_label_3.csv', index=False)
    print("Saved successfully to CSV")


create_csv(y_pred_test_label3)
