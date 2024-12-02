import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('user_behavior_dataset.csv')

# Data Cleaning (Check for duplicates)
df.drop_duplicates(inplace=True)

# Exploratory Data Analysis (EDA)
# 1. Distribution of App Usage Time
plt.figure(figsize=(8, 5))
sns.histplot(df['App Usage Time (min/day)'], kde=True, bins=30)
plt.title('Distribution of App Usage Time (min/day)')
plt.xlabel('App Usage Time (min/day)')
plt.show()

# 2. Data Usage by Age and Gender
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Data Usage (MB/day)', data=df)
plt.title('Data Usage by Gender')
plt.show()

# 3. Number of Apps Installed by Operating System
plt.figure(figsize=(10, 5))
sns.boxplot(x='Operating System', y='Number of Apps Installed', data=df)
plt.title('Number of Apps Installed by Operating System')
plt.show()

# 4. Battery Drain by Screen On Time
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Screen On Time (hours/day)', y='Battery Drain (mAh/day)', hue='Operating System', data=df)
plt.title('Battery Drain vs Screen On Time')
plt.show()

# 5. Correlation Heatmap (New Visualization)
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(12, 8))
corr = numeric_df.corr()  # Compute correlation only on numeric data
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
# Feature Selection and Encoding
X = df.drop(columns=['User ID', 'User Behavior Class'])
X = pd.get_dummies(X, drop_first=True)
y = df['User Behavior Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print('Classification Report:')
print(classification_report(y_test, y_pred))

print('Confusion Matrix:')
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
