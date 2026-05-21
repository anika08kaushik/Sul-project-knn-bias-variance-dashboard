import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

# Read dataset

df = pd.read_csv('heart.csv')

# =========================
# BASIC DATASET CHECKS
# =========================

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# =========================
# CHECK MISSING VALUES
# =========================

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# REMOVE DUPLICATES
# =========================

print("\nDuplicate Rows:", df.duplicated().sum())

# Drop duplicates

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates:")
print(df.shape)

# =========================
# DATASET SUMMARY
# =========================

print("\nStatistical Summary:")
print(df.describe())

# =========================
# CLASS DISTRIBUTION
# =========================

print("\nTarget Class Distribution:")
print(df['target'].value_counts())

# Plot class distribution

plt.figure(figsize=(5,4))
df['target'].value_counts().plot(kind='bar')
plt.title('Heart Disease Class Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# =========================
# FEATURE/TARGET SPLIT
# =========================

X = df.drop('target', axis=1)
y = df['target']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN KNN FOR DIFFERENT K
# =========================

k_values = [1, 3, 5, 7, 11, 15, 21]

train_acc = []
test_acc = []

print("\nTraining Models...\n")

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    train_acc.append(train_accuracy)
    test_acc.append(test_accuracy)

    print(f"K = {k}")
    print(f"Training Accuracy = {train_accuracy:.4f}")
    print(f"Testing Accuracy = {test_accuracy:.4f}")
    print("-" * 40)

# =========================
# FINAL GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(k_values, train_acc, marker='o', label='Training Accuracy')
plt.plot(k_values, test_acc, marker='o', label='Testing Accuracy')

plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('Effect of K on Bias-Variance Tradeoff')

plt.legend()
plt.grid(True)

plt.savefig('knn_results.png')

plt.show()

print("\nProject Completed Successfully!")