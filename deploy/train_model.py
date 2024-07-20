import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle TotalCharges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Define features and target
target_column = 'Churn_Yes' if 'Churn_Yes' in df.columns else 'Churn'
feature_columns_to_drop = ['customerID', target_column] if 'customerID' in df.columns else [target_column]

X = df.drop(columns=feature_columns_to_drop)
y = df[target_column]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model and the scaler
with open('churn_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
with open('columns.pkl', 'wb') as columns_file:
    pickle.dump(X.columns.to_list(), columns_file)

print("Model training completed and saved as churn_model.pkl")
