# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Set pandas display options for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load training and holdout datasets
df_data = pd.read_csv('/Users/han/Desktop/Humana data comp/holdout_model/train_dataset/XGB_train.csv')
df_ho_data = pd.read_csv('/Users/han/Desktop/Humana data comp/holdout_model/train_dataset/XGB_holdout.csv')

# Define features (X) and target (y)
X = df_data.drop(columns=['id', 'preventive_visit_gap_ind'])
y = df_data['preventive_visit_gap_ind']

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ——————————————————————————————————————————————————————————————————————————————————————————
# XGBoost model (using best paramters based on grid search + random search)

# Define the best parameters for the XGBoost model
best_params = {
    'colsample_bytree': 0.7098887171960256,
    'gamma': 0.2806217129238506,
    'learning_rate': 0.12487806242613694,
    'max_depth': 9,
    'n_estimators': 180,
    'subsample': 0.9043140194467589,
    'objective': 'binary:logistic',  # Binary classification
    'eval_metric': 'logloss',        # Evaluation metric
    'random_state': 42
}

# Initialize the XGBoost model with the best parameters
best_model = xgb.XGBClassifier(**best_params)

# Train the model on the training data
best_model.fit(X_train, y_train)

# Predict on the test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.6f}")
print(f"ROC-AUC Score: {roc_auc:.6f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='black', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='forestgreen', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Early stopping and evaluation during training
eval_set = [(X_train, y_train), (X_test, y_test)]
best_model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=20, verbose=True)

# Visualize training and testing log loss over epochs
train_logloss = best_model.evals_result()['validation_0']['logloss']
test_logloss = best_model.evals_result()['validation_1']['logloss']
epochs = range(len(train_logloss))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_logloss, label='Train Log Loss', color='darkseagreen')
plt.plot(epochs, test_logloss, label='Test Log Loss', color='skyblue')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss Over Epochs')
plt.legend()
plt.show()

# ——————————————————————————————————————————————————————————————————————————————————————
# Feature Importance

feature_names = X.columns
feature_importance = best_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.head(5))

# Visualize top feature importance
top_n = 5
top_features_df = feature_importance_df.head(top_n)
plt.figure(figsize=(12, 8))
plt.barh(top_features_df['Feature'], top_features_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title(f'Top {top_n} Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# ——————————————————————————————————————————————————————————————————————————————————————
# XGBoost (Grid Search)

# # Uncomment and use this section to fine-tune hyperparameters using GridSearchCV
# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     max_depth=9,
#     eval_metric='logloss',
#     random_state=42
# )
# param_grid = {
#     'colsample_bytree': [0.69, 0.71, 0.73],
#     'gamma': [0.26, 0.28, 0.30],
#     'learning_rate': [0.12, 0.125, 0.13],
#     'n_estimators': [170, 180, 190],
#     'subsample': [0.89, 0.90, 0.91],
#     "reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1, 100],
#     "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 100],
#     "min_child_weight": [2, 3, 4, 5, 6, 7, 8]
# }
# grid_search = GridSearchCV(
#     estimator=xgb_model,
#     param_grid=param_grid,
#     scoring='roc_auc',
#     cv=2,
#     n_jobs=1,
#     verbose=1
# )
# grid_search.fit(X_train, y_train)
# print("Best Parameters:", grid_search.best_params_)
# print("Best ROC-AUC Score:", grid_search.best_score_)
# best_model_grid = grid_search.best_estimator_

# ——————————————————————————————————————————————————————————————————————————————————————
# Predict Holdout Data

ids = df_ho_data['id']
X_holdout = df_ho_data.drop(columns=['id'])

# Predict probabilities on the holdout dataset
scores = best_model.predict_proba(X_holdout)[:, 1]

# Create a results DataFrame with ID, score, and rank
results = pd.DataFrame({'id': ids, 'SCORE': scores})
results['RANK'] = results['SCORE'].rank(ascending=False, method='dense').astype('int')

# Save the holdout results to a CSV file
results.to_csv('2024CaseCompetition_Han_Bao_20241010.csv', index=False)

print("Results saved successfully.")
