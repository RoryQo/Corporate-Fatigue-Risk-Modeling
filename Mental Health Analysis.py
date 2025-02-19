#!/usr/bin/env python
# coding: utf-8

# # Burn Out Prediction: Mental Health Analysis

# ### Setup

# In[69]:


# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SequentialFeatureSelector, RFE, SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import random
import xgboost as xgb
import graphviz
from graphviz import Source, Digraph
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
from sklearn.inspection import permutation_importance
import shap


# In[70]:


# Define the folder path
path = "C://Users//roryq//Downloads//MQE_Data//"  # Replace with your actual folder path

# Input and output file names
input_file = f"{path}Stress.csv"

df = pd.read_csv(input_file, low_memory=False)

# Set a global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# In[71]:


df.head()


# In[72]:


df.info()


# In[73]:


df['Burnout_Symptoms'].unique()


# In[74]:


df['Company_Size'].unique()


# In[75]:


#  Inspect and explore categorical (or string) columns
# Loop through columns with dtype 'object' (categorical columns)
for col in df.select_dtypes(include='object'):
    print(df[col].unique())


# ### Data Exploration

# In[76]:


# Set up the figure and axis for the grid
num_columns = len(df.select_dtypes(include=['number']).columns)
num_rows = (num_columns // 4) + (num_columns % 4 > 0)  # Set 3 columns per row, calculate the necessary rows
fig, axes = plt.subplots(num_rows, 4, figsize=(18, num_rows * 3))  # Adjust the figsize as needed

# Flatten the axes array to easily iterate over it
axes = axes.flatten()

# Loop over the numeric columns and plot the histograms
for idx, col in enumerate(df.select_dtypes(include=['number']).columns):
    axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

# Remove empty subplots (if any)
for idx in range(num_columns, len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# In[77]:


df['Burnout_Symptoms'].value_counts()


# In[78]:


df_encoded = pd.get_dummies(df, drop_first=True)
# Compute Correlation Matrix (Only for Numerical Features)
correlation_matrix = df.corr(method='spearman',numeric_only=True)
# Plot Heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[79]:


missing_percentage = (df.isnull().sum() / len(df)) * 100

# To handle NaN values in the missing percentage (e.g., if a column is entirely NaN)
missing_percentage = missing_percentage.fillna(0)

print("\nPercentage of missing values in each column:")
print(missing_percentage)


# In[80]:


df['Health_Issues'].value_counts()


# In[81]:


import pandas as pd
import numpy as np
import scipy.stats as ss

def cramers_v(x, y):
    """Computes Cramér's V statistic for categorical-categorical correlation."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((k_corr-1), (r_corr-1)))

# Compute correlation
correlation = cramers_v(df["Health_Issues"], df["Burnout_Symptoms"])
print(f"Cramér's V Correlation between Health_Issues and Burnout_Symptoms: {correlation:.3f}")


# ### Data Wrangling

# Imputation
# Variable encoding
# attempted knn imputation, with low accuracy indicating there was not a pattern in missingness, so mode impuptation was final decision

# In[82]:


# Those that have NA values in health have no health problems
# Encode No in place of NAs
df["Health_Issues"].fillna("No", inplace=True)


# In[83]:


missing_percentage = (df.isnull().sum() / len(df)) * 100

# To handle NaN values in the missing percentage (e.g., if a column is entirely NaN)
missing_percentage = missing_percentage.fillna(0)

print("\nPercentage of missing values in each column:")
print(missing_percentage)


# In[84]:


# Label encode bc size is ordinal 

#Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode a specific column (e.g., "Category_Column")
df["Company_Size"] = label_encoder.fit_transform(df["Company_Size"])


# In[85]:


# For binary encoding include the occasional symptoms to the yes category
# Reducing Dimensionality
df["Burnout_Symptoms"] = df["Burnout_Symptoms"].replace("Occasional", "Yes")

df["Health_Issues"] = df["Health_Issues"].replace("Both", "Yes")
df["Health_Issues"] = df["Health_Issues"].replace("Physical", "Yes")
df["Health_Issues"] = df["Health_Issues"].replace("Mental", "Yes")


# In[86]:


# One Hot encode the rest

# Identify categorical columns (excluding numeric ones)
categorical_cols = df.select_dtypes(include=["object"]).columns

df[categorical_cols] = df[categorical_cols].fillna("Unknown")  # Replace NaNs
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df.info()


# ### Modeling

# In[87]:


# Selecting Features and Targets

X = df.drop(columns=["Burnout_Symptoms_Yes"])

y = df["Burnout_Symptoms_Yes"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features (important for SVM and Neural Networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:





# #### Neural Network

# #### The Nueral Network optimized with accuracy score, only predicted burnout symptoms. We continued to optimize using f1 score; grid is artificially shrunk from original after initial run so that it is easier for grader to run

# In[88]:


# Define hyperparameters to tune
param_grid = {
    'hidden_layer_sizes': [(64,), (128,)],  # Different layer architectures
    'activation': ['relu'],  # Try different activation functions
    'solver': ['adam'],  # Optimizers: Adam vs. SGD
    'alpha': [0.0001],  # Regularization strength
    'learning_rate': ['constant', ]  # Learning rate strategies
}

# Initialize MLP Classifier
mlp = MLPClassifier(max_iter=200, random_state=42)

# Grid Search with Cross-Validation (2-fold)
grid_search = GridSearchCV(mlp, param_grid, cv=2, scoring='f1_weighted', n_jobs=-1, verbose=1)

# Fit the model on training data
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print(f" Best Hyperparameters: {grid_search.best_params_}")
print(f" Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")


# In[89]:


# Get the best model from GridSearch
best_mlp = grid_search.best_estimator_

# Predict on the test set
y_pred_best_mlp = best_mlp.predict(X_test)

# Evaluate performance
from sklearn.metrics import accuracy_score, classification_report

print(" Best Neural Network Performance:")
print(classification_report(y_test, y_pred_best_mlp))

# Final accuracy on test data
final_accuracy = accuracy_score(y_test, y_pred_best_mlp)
print(f" Final Neural Network Accuracy on Test Set: {final_accuracy:.4f}")


# In[90]:


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_best_mlp)

# Display the confusion matrix in blue
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)  # Use the 'Blues' colormap

# Show the plot
plt.show()


# #### Random Forest

# In[91]:


# Initialize the Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [ 100],  # Number of trees in the forest
    'max_depth': [ None],  # Maximum depth of each tree
    'min_samples_split': [2],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1]  # Minimum number of samples required to be at a leaf node
}

# Grid Search with Cross-Validation
rf_grid_search = GridSearchCV(rf, rf_param_grid, scoring='f1_weighted', verbose=1, cv=2)

# Fit the model using GridSearchCV
rf_grid_search.fit(X_train, y_train)

# Get the best Random Forest model
best_rf = rf_grid_search.best_estimator_

# Make predictions on the test set
y_pred_rf = best_rf.predict(X_test)

# Evaluate the Random Forest model
print(" Random Forest Classifier Performance:")
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")

# Print best parameters and best score
print(f" Best Hyperparameters: {rf_grid_search.best_params_}")
print(f" Best Cross-Validation Accuracy: {rf_grid_search.best_score_:.4f}")


# In[92]:


# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp_rf.plot(cmap=plt.cm.Blues)
plt.show()


# #### KNN

# In[93]:


# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Define hyperparameter grid for KNN
knn_param_grid = {
    'n_neighbors': [3, 5, 10],  # Number of neighbors to consider
    'weights': ['uniform', 'distance'],  # Weighting function for prediction
    'metric': ['euclidean']  # Distance metric
}

# Grid Search with Cross-Validation
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Fit the model using GridSearchCV
knn_grid_search.fit(X_train, y_train)

# Get the best KNN model
best_knn = knn_grid_search.best_estimator_

# Make predictions on the test set
y_pred_knn = best_knn.predict(X_test)

# Evaluate the KNN model
print(" KNN Classifier Performance:")
print(classification_report(y_test, y_pred_knn))
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}\n")
# Print best parameters and best score
print(f" Best Hyperparameters: {knn_grid_search.best_params_}")
print(f" Best Cross-Validation Accuracy: {knn_grid_search.best_score_:.4f}")


# In[94]:


# Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap=plt.cm.Blues)
plt.show()


# #### Naive Bayes

# In[95]:


# Initialize the Naive Bayes classifier
nb = GaussianNB()

# Since Gaussian Naive Bayes doesn't have many hyperparameters to tune, we usually don't need GridSearchCV
# But, let's show a simple example of tuning the 'var_smoothing' parameter, which helps with numerical stability
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
}

# Grid Search with Cross-Validation (if you want to fine-tune the var_smoothing parameter)
nb_grid_search = GridSearchCV(nb, param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Fit the model using GridSearchCV
nb_grid_search.fit(X_train, y_train)

# Get the best Naive Bayes model
best_nb = nb_grid_search.best_estimator_

# Make predictions on the test set
y_pred_nb = best_nb.predict(X_test)

# Evaluate the Naive Bayes model
print("  Naive Bayes Classifier Performance:")
print(classification_report(y_test, y_pred_nb))
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\n")

# Print best parameters and best score
print(f"  Best Hyperparameters: {nb_grid_search.best_params_}")
print(f"  Best Cross-Validation Accuracy: {nb_grid_search.best_score_:.4f}")


# In[96]:


# Confusion Matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
disp_nb.plot(cmap=plt.cm.Blues)
plt.show()


# #### XGBoost

# In[97]:


# Initialize the XGBoost classifier
xgboost_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# Define hyperparameter grid for XGBoost
xgboost_param_grid = {
    'n_estimators': [200],  # Number of trees in the model
    'learning_rate': [0.1, 0.2],  # Step size for updating weights
    'max_depth': [3, 7],  # Maximum depth of each tree 
    'colsample_bytree': [0.5, 0.8],  # Fraction of features to use for each tree
    'gamma': [0, 0.2]  # Regularization parameter for pruning
}

# Grid Search with Cross-Validation
xgboost_grid_search = GridSearchCV(xgboost_model, xgboost_param_grid, cv=5, scoring='f1_weighted', verbose=1)

# Fit the model using GridSearchCV
xgboost_grid_search.fit(X_train, y_train)

# Get the best XGBoost model
best_xgboost = xgboost_grid_search.best_estimator_

# Make predictions on the test set
y_pred_xgboost = best_xgboost.predict(X_test)

# Evaluate the XGBoost model
print(" XGBoost Classifier Performance:")
print(classification_report(y_test, y_pred_xgboost))
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgboost):.4f}\n")

# Print best parameters and best score
print(f" Best Hyperparameters: {xgboost_grid_search.best_params_}")
print(f" Best Cross-Validation Accuracy: {xgboost_grid_search.best_score_:.4f}")



# In[98]:


# Confusion Matrix
cm_xgboost = confusion_matrix(y_test, y_pred_xgboost)
disp_xgboost = ConfusionMatrixDisplay(confusion_matrix=cm_xgboost)
disp_xgboost.plot(cmap=plt.cm.Blues)
plt.show()


# #### SVM

# In[99]:


# Initialize and train the SVM classifier without GridSearchCV
best_svm = SVC(C=0.1, kernel='linear', gamma='scale', random_state=42)

# Fit the model on training data
best_svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = best_svm.predict(X_test)

# Evaluate the SVM model
print("SVM Classifier Performance:")
print(classification_report(y_test, y_pred_svm))
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")


# ### Model Comparisons

# In[100]:


# Dictionary to store computation times
computation_times = {}



# List of models and their names
models = {
    
 
    "KNN": best_knn,
    "MLP": best_mlp,
    "XGBoost": best_xgboost,
    "Random Forest": best_rf,
    "Naive Bayes": best_nb,
    "SVM": best_svm
}

# Perform 5-fold cross-validation and collect scores
cv_results = {}
for name, model in models.items():
    start_time = time.time()
    
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1)
    cv_results[name] = scores
   
    end_time = time.time()  # End timing
    
    computation_times[name] = end_time - start_time  # Store elapsed time



# In[101]:


cv_results


# In[102]:


# Extract averages, min, and max values
model_names = list(cv_results.keys())
avg_scores = [np.mean(scores) for scores in cv_results.values()]
min_scores = [np.min(scores) for scores in cv_results.values()]
max_scores = [np.max(scores) for scores in cv_results.values()]

# Compute overlapping error regions (min-max ranges)
overall_min = np.max(min_scores)  # Max of the minimums
overall_max = np.min(max_scores)  # Min of the maximums


# Plot the results
plt.figure(figsize=(10, 5))
plt.bar(model_names, avg_scores, yerr=[np.array(avg_scores) - np.array(min_scores), np.array(max_scores) - np.array(avg_scores)], capsize=5, color='blue', alpha=0.4)

# Highlight overlap region with a red band
if overall_min < overall_max:
    plt.axhspan(overall_min, overall_max, color='red', alpha=0.3, label="Overlapping Error Range")
plt.ylabel("Cross-Validated F1 Weighted Score")
plt.title("Model Performance with 5-Fold Cross-Validation")
plt.ylim(0.5, 0.58)  # Accuracy ranges from 0 to 1
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)

plt.annotate("Note: Error bars represent the minimum and maximum\n"
             "F1 scores across 5 cross-validation folds.",
             xy=(0.5, -0.4), xycoords='axes fraction', 
             ha='center', fontsize=10, color="black")

plt.show()


# Initially it appeared simple classifiers had a low accuracy so we continued to fit more complex models, however after comparing model preformance we can see that that initial simple classifiers were the best overall

# In[103]:


# Extract model names and their computation times
model_names = list(computation_times.keys())
comp_times = list(computation_times.values())


# Sort models by computation time (ascending order)
sorted_indices = np.argsort(comp_times)
sorted_model_names = np.array(model_names)[sorted_indices]
sorted_comp_times = np.array(comp_times)[sorted_indices]

# Plot Computation Time Comparison (Ordered by Time)
plt.figure(figsize=(10, 5))
plt.bar(sorted_model_names, sorted_comp_times, color='green', alpha=0.6)
plt.ylabel("Computation Time (seconds)")
plt.title("Computation Time for 5-Fold Cross-Validation")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Annotate the bars with exact time values
for i, v in enumerate(sorted_comp_times):
    plt.text(i, v + 0.2, f"{v:.2f}s", ha='center', fontsize=10, color='black')

plt.show()


# #### Features and Gender

# In[104]:


# Get feature importances from XGBoost
feature_importances = best_xgboost.feature_importances_
feature_names = X.columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_names = np.array(feature_names)[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Plot Feature Importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align="center", color='skyblue')
plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.title("Feature Importance in XGBoost Model")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.savefig("feature_importance_XG.png", dpi=300, bbox_inches="tight")  # Save as PNG


# In[105]:


# Compute permutation importance
result = permutation_importance(best_mlp, X_test, y_test, scoring='f1_weighted', n_repeats=2, random_state=42)

# Extract feature importance scores
feature_importances = result.importances_mean
feature_names = X.columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_names = np.array(feature_names)[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Plot Feature Importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align="center", color='skyblue')
plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score (Permutation)")
plt.title("Feature Importance in MLP Model (Permutation Importance)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[106]:


# Compute permutation importance
result = permutation_importance(best_knn, X_test, y_test, scoring='f1_weighted', n_repeats=5, random_state=42)

# Extract feature importance scores
feature_importances = result.importances_mean
feature_names = X.columns

# Sort features by importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_names = np.array(feature_names)[sorted_indices]
sorted_importances = feature_importances[sorted_indices]

# Plot Feature Importance
plt.figure(figsize=(12, 6))
plt.bar(range(len(sorted_importances)), sorted_importances, align="center", color='skyblue')
plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance Score (Permutation)")
plt.title("Feature Importance in Knn Model (Permutation Importance)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[107]:


# Define the specific features you want to plot
top_features = ["Job_Satisfaction", "Experience_Years", "Physical_Activity_Hours_per_Week", 
                     "Marital_Status_Widowed", "Annual_Leaves_Taken", "Health_Issues_Yes", 
                     "Department_Sales", "Department_Sales", "Manager_Support_Level", "Department_Finance"]


# Replace boolean values in "Gender_Male" with "Female" and "Male"
df["Gender_Label"] = df["Gender_Male"].replace({0: "Female", 1: "Male"})

# Convert categorical variables back for grouping
df_grouped = df.groupby(["Gender_Label"])[top_features].mean().reset_index()

# Create a grid layout for subplots
fig, axes = plt.subplots(4, 3, figsize=(20, 20))  # 4 rows, 3 columns for 10 features
axes = axes.flatten()

# Plot differences in top features across Gender
for i, feature in enumerate(top_features):
    sns.barplot(data=df, x="Gender_Label", y=feature, palette="coolwarm", ax=axes[i])
    axes[i].set_title(f"{feature} by Gender")
    axes[i].set_xlabel("Gender")
    axes[i].set_ylabel(f"Mean {feature}")

# Remove empty subplots if less than grid size
for i in range(len(top_features), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout for readability
plt.tight_layout()
plt.show()


# In[108]:


# Convert categorical variables back for grouping
df_grouped = df.groupby(["Remote_Work"])[top_features].mean().reset_index()

# Create a grid layout for subplots
fig, axes = plt.subplots(4, 3, figsize=(20, 20))  # 2 rows, 3 columns
axes = axes.flatten()

# Plot differences in top features across Gender & Remote Work
for i, feature in enumerate(top_features):
    sns.barplot(data=df, x="Remote_Work", y=feature,  palette="coolwarm", ax=axes[i])
    axes[i].set_title(f"{feature} by Remote Work")
    axes[i].set_xlabel("Remote Work")
    axes[i].set_ylabel(f"Mean {feature}")
    
# Remove empty subplot (if number of features is less than grid size)
for i in range(len(top_features), len(axes)):
    fig.delaxes(axes[i])

# Adjust layout
plt.tight_layout()
plt.show()


# In[109]:


import matplotlib.pyplot as plt

# Count total men and women in the dataset
gender_counts = df["Gender_Male"].value_counts()

# Count the number of burnout cases by gender
burnout_counts = df.groupby("Gender_Male")["Burnout_Symptoms_Yes"].sum()

# Compute proportion of burnout cases per gender
burnout_proportions = burnout_counts / gender_counts

# Plot bar chart
plt.figure(figsize=(8, 5))
plt.bar(burnout_proportions.index, burnout_proportions.values, color=["skyblue", "seagreen"])
plt.xlabel("Gender")
plt.ylabel("Proportion of Burnout Cases")
plt.title("Proportion of Individuals with Burnout Symptoms by Gender")
plt.ylim(0,1)
# Show values on bars
for i, v in enumerate(burnout_proportions.values):
    plt.text(i, v + 0.01, f"{v:.2%}", ha='center', fontsize=12)

plt.show()


# ### Reduction in Top Predictors

# In[110]:


# Ensure feature names are retrieved BEFORE preprocessing (replace df with your original dataset)
original_feature_names = df.drop(columns=["Burnout_Symptoms_Yes"]).columns.tolist()

# Convert X_test back to a DataFrame with original column names
X_test = pd.DataFrame(X_test, columns=original_feature_names[:X_test.shape[1]])

# Check that feature names were restored
print("Updated Feature Names in X_test:\n", X_test.columns.tolist())

# Create a copy for policy simulation
X_test_policy = X_test.copy()
X_test_policy


# In[111]:


# Ensure predictors exist in dataset
top_predictors = ["Commute_Time_Hours", "Experience_Years"]  # Replace with actual feature names

# Apply a 20% reduction for non-binary features
for feature in top_predictors:
    if feature in X_test_policy.columns:
        # Reduce all values by 20% proportionally
        X_test_policy[feature] *= 0.8
        print(f"✅ Reduced {feature} by 20%")

# Print before and after average values
for feature in top_predictors:
    before = X_test[feature].mean()
    after = X_test_policy[feature].mean()
    print(f"{feature} - Before: {before:.4f}, After: {after:.4f}")


# In[112]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# Predict burnout before and after policy
y_pred_original = best_knn.predict(X_test)
y_pred_policy = best_knn.predict(X_test_policy)

# Evaluate effectiveness
original_burnout_rate = np.mean(y_pred_original)
policy_burnout_rate = np.mean(y_pred_policy)
reduction_percentage = (original_burnout_rate - policy_burnout_rate) / original_burnout_rate * 100 if original_burnout_rate > 0 else 0

# Print evaluation results
print("Original Model Performance:")
print(classification_report(y_test, y_pred_original))
print(f"Original Predicted Burnout Rate: {original_burnout_rate:.4f}\n")

print("Policy-Adjusted Model Performance:")
print(classification_report(y_test, y_pred_policy))
print(f"Predicted Burnout Rate After Policy: {policy_burnout_rate:.4f}\n")

print(f"Estimated Reduction in Burnout: {reduction_percentage:.2f}%")

# Visualize burnout rate before and after policy
plt.figure(figsize=(6, 4))
plt.bar(["Original", "Policy Adjusted"], [original_burnout_rate, policy_burnout_rate], color=["red", "green"])
plt.xlabel("Scenario")
plt.ylabel("Predicted Burnout Rate")
plt.title("Effect of Policy on Predicted Burnout Rate")
plt.ylim(0.5, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.show()


# In[ ]:





# In[113]:


# Check available columns
print(X_test.columns.tolist())

# Check if the column exists
if top_predictors[0] not in X_test.columns:
    print(f"Column {top_predictors[0]} not found in X_test!")
if top_predictors[1] not in X_test.columns:
    print(f"Column {top_predictors[1]} not found in X_test!")


# In[114]:


print(f"Original {top_predictors[0]} count: {X_test[top_predictors[0]].sum()}")
print(f"Modified {top_predictors[0]} count: {X_test_policy[top_predictors[0]].sum()}")

print(f"Original {top_predictors[1]} count: {X_test[top_predictors[1]].sum()}")
print(f"Modified {top_predictors[1]} count: {X_test_policy[top_predictors[1]].sum()}")


# In[115]:


print("Checking differences in features after policy change...")
print(X_test[top_predictors].describe())
print(X_test_policy[top_predictors].describe())

print("\nChecking prediction distributions:")
print("Original:", np.bincount(y_pred_original))
print("Policy:", np.bincount(y_pred_policy))


# In[116]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Predict burnout before and after policy change
y_pred_original = best_knn.predict(X_test)
y_pred_policy = best_knn.predict(X_test_policy)

# Compute confusion matrices
cm_original = confusion_matrix(y_test, y_pred_original)
cm_policy = confusion_matrix(y_test, y_pred_policy)

# Plot the confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original Confusion Matrix
sns.heatmap(cm_original, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
axes[0].set_title("Original Model Confusion Matrix")
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

# Policy-Adjusted Confusion Matrix
sns.heatmap(cm_policy, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1])
axes[1].set_title("Policy-Adjusted Confusion Matrix")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# Show plots
plt.tight_layout()
plt.show()

# Print classification reports for comparison
print("Original Model Performance:")
print(classification_report(y_test, y_pred_original))

print("\nPolicy-Adjusted Model Performance:")
print(classification_report(y_test, y_pred_policy))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# Calculate the proportion of burnout predictions before and after policy
original_burnout_rate = np.mean(y_pred_original)
policy_burnout_rate = np.mean(y_pred_policy)

# Calculate the absolute difference in proportions
difference = original_burnout_rate - policy_burnout_rate

# Plot the difference in proportion of burnout prediction
plt.figure(figsize=(6, 4))
plt.bar(["Original", "Policy Adjusted"], [original_burnout_rate, policy_burnout_rate], color=["blue", "green"])
plt.xlabel("Scenario")
plt.ylabel("Proportion of Burnout Predictions")
plt.title("Change in Burnout Prediction Before and After Policy")
plt.ylim(.785, .792)
plt.grid(axis='y', linestyle='--', alpha=0.7)
775
# Annotate the difference
plt.text(0.5, (original_burnout_rate + policy_burnout_rate) / 2, f"Δ = {difference:.2%}", 
         ha='center', va='center', fontsize=12, color='black', fontweight='bold')

# Show the plot
plt.show()


# In[ ]:




