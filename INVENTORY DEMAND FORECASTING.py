# INVENTORY-DEMAND-FORECASTING
Developed a predictive model to forecast product demand by category using sales data from Amazon. The project aimed to optimize inventory management and reduce holding costs by identifying high-demand products.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("/content/Amazon 2_Raw.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Summary statistics of numerical columns
print("\nSummary statistics of numerical columns:")
print(data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Unique values in categorical columns
print("\nUnique values in categorical columns:")
print(data.nunique())



# Distribution of sales quantity and profit
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data['Quantity'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sales Quantity')

plt.subplot(1, 2, 2)
sns.histplot(data['Profit'], bins=20, kde=True, color='salmon')
plt.title('Distribution of Profit')

plt.tight_layout()
plt.show()

# Time series analysis (if applicable)
if 'Order Date' in data.columns:
    data['Order Date'] = pd.to_datetime(data['Order Date'])
    data.set_index('Order Date', inplace=True)
    plt.figure(figsize=(12, 6))
    data['Sales'].resample('M').sum().plot()
    plt.title('Monthly Sales')
    plt.ylabel('Sales')
    plt.xlabel('Date')
    plt.show()
import pandas as pd

# Load the dataset
data = pd.read_csv("/content/Amazon 2_Raw.csv")

# Display the columns of the dataset
print("Columns of the dataset:")
print(data.columns)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("/content/Amazon 2_Raw.csv")

# Define the threshold for high demand based on sales volume
threshold = 200  # Example threshold for defining high demand

# Create the target variable 'Demand' based on sales volume
data['Demand'] = data['Sales'].apply(lambda x: 1 if x > threshold else 0)

# Select relevant features and define the target variable
X = data[['Sales', 'Quantity', 'Profit']]  # Example features
y = data['Demand']  # Target variable (binary: 1 for high demand, 0 for low demand)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Load the new dataset
new_data = pd.read_csv("/content/Amazon 2_Raw.csv")

# Assuming X_new is the feature matrix of the new dataset for which you want to predict demand
# Select relevant features
X_new = new_data[['Sales', 'Quantity', 'Profit']]

# Standardize features
X_new_scaled = scaler.transform(X_new)

# Make predictions on the new dataset
predicted_demand = model.predict(X_new_scaled)

# Add predicted demand as a new column in the new dataset
new_data['Predicted_Demand'] = predicted_demand

# Filter products predicted to have high demand
high_demand_products = new_data[new_data['Predicted_Demand'] == 1]['Product Name']

# Display the products predicted to have high demand
print("Products predicted to have high demand:")
print(high_demand_products)
import matplotlib.pyplot as plt

# Histogram of Sales
plt.figure(figsize=(10, 6))
plt.hist(data['Sales'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()
import matplotlib.pyplot as plt

# Filter products predicted to have high demand
high_demand_products = new_data[new_data['Predicted_Demand'] == 1]

# Count the number of products predicted to have high demand per category
high_demand_counts = high_demand_products['Category'].value_counts()

# Create a bar chart
plt.figure(figsize=(10, 6))
high_demand_counts.plot(kind='bar', color='skyblue')
plt.title('Count of Products Predicted to Have High Demand by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
