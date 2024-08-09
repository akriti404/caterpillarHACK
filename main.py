import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Problem Statement 2_ Data set.xlsx - Data Set.csv')


# Display the first few rows of the dataset
print(data.head())

# Get basic statistics and information about the dataset
print(data.info())
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Step 3: Data Preprocessing
# Handle missing values (if any)
data = data.dropna()  # Alternatively, you might want to impute missing values

# Convert categorical variables to numerical (if applicable)
data = pd.get_dummies(data)


# Separate features and target variable
X = data.drop(data.columns[2], axis=1)  # Replace 'target_column' with the actual column name
y = data.iloc[:,2]

# Step 4: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train a Bayesian Classifier
# Initialize the model (you can choose among GaussianNB, MultinomialNB, BernoulliNB)
model = GaussianNB()  # You can switch to MultinomialNB() or BernoulliNB() depending on your data type

# Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 7: Visualize the Confusion Matrix (Optional)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 8: Interpret Results and Make Recommendations
# Analyze the performance and provide actionable insights