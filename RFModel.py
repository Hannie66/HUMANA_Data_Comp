# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Set pandas display options for better data visualization
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# Load training data
data = pd.read_csv('/Users/han/Desktop/Humana data comp/Final_Train_Data.csv')
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix for numeric features
corr_matrix = numeric_df.corr()

# Identify highly correlated features (correlation > threshold) and drop them
threshold = 0.9
columns_to_drop = set()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]  # Get column name
            columns_to_drop.add(colname)

print(f"Columns to drop due to high correlation: {columns_to_drop}")

# Create a reduced dataframe without highly correlated features
df_reduced = df.drop(columns=columns_to_drop)

# Define features (X) and target variable (y)
X = df.drop(['id', 'preventive_visit_gap_ind'], axis=1)
y = df['preventive_visit_gap_ind']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=77)

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Load holdout data
holdout_data = pd.read_csv('/Users/han/Desktop/Humana data comp/hold_out_clean/ho_final_clean.csv')
holdout_df = pd.DataFrame(holdout_data)

# Ensure the holdout data has the same number of features as the training data
n_features = X_train.shape[1]
holdout_df_aligned = holdout_df.iloc[:, :n_features]

print("\nNumber of features in training data:", n_features)
print("Holdout data aligned with training features.")
