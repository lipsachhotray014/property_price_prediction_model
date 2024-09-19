# Let's load the dataset from the provided file to inspect its structure and see how to apply the problem statement.
import pandas as pd

# Load the uploaded Excel file
file_path = '/Users/lipsachhotray/Desktop/MLProcessing.xlsx'
data = pd.read_excel(file_path)

# Remove $ symbol from all columns and convert to numeric where applicable
data = data.replace({'\$': ''}, regex=True)

# Step 1: Handle missing values and check for any inconsistencies
missing_values = data.isnull().sum()

# Display the columns with missing values, if any
missing_values[missing_values > 0]

# Attempt to convert all columns to numeric where possible
data = data.apply(pd.to_numeric, errors='ignore')

# Handle missing values in numerical columns by imputing the median
numerical_columns = ['room_bed', 'room_bath', 'living_measure', 'lot_measure', 'ceil', 'sight',
                     'condition', 'living_measure15', 'lot_measure15', 'total_area']

for column in numerical_columns:
    data[column].fillna(data[column].median(), inplace=True)

# Handle missing values in the categorical 'furnished' column by imputing the mode (most frequent value)
data['furnished'].fillna(data['furnished'].mode()[0], inplace=True)

# Check again to ensure no missing values remain
missing_values_post_imputation = data.isnull().sum()
missing_values_post_imputation[missing_values_post_imputation > 0]

# Ensure the dayhours column is in datetime format
data['dayhours'] = pd.to_datetime(data['dayhours'], errors='coerce')

# Create new columns for year, month, and day
data['sale_year'] = data['dayhours'].dt.year
data['sale_month'] = data['dayhours'].dt.month
data['sale_day'] = data['dayhours'].dt.day

# Display the first few rows to verify the transformation
data[['dayhours', 'sale_year', 'sale_month', 'sale_day']].head()


# One-hot encode the zipcode column
data = pd.get_dummies(data, columns=['zipcode'], prefix='zipcode', drop_first=True)

from sklearn.preprocessing import StandardScaler

# Features with large ranges
scale_columns = ['living_measure', 'lot_measure', 'total_area']

# Apply standardization (zero mean, unit variance)
scaler = StandardScaler()
data[scale_columns] = scaler.fit_transform(data[scale_columns])

# Correlation matrix for all features
correlation_matrix = data.corr()

# Display correlation with target variable 'price'
correlation_with_price = correlation_matrix['price'].sort_values(ascending=False)

# Visualize correlation matrix using seaborn heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
# plt.show()


# Visualize distribution of key features
sns.histplot(data['price'], kde=True).set_title('Price Distribution')
# plt.show()

sns.histplot(data['living_measure'], kde=True).set_title('Living Measure Distribution')
# plt.show()

sns.histplot(data['room_bed'], kde=True).set_title('Number of Bedrooms Distribution')
# plt.show()

sns.histplot(data['quality'], kde=True).set_title('Quality Distribution')
# plt.show()

# Boxplot for outlier detection
sns.boxplot(x=data['price']).set_title('Price Boxplot')
# plt.show()

sns.boxplot(x=data['living_measure']).set_title('Living Measure Boxplot')
# plt.show()

# Create interaction feature: condition × quality
data['condition_quality_interaction'] = data['condition'] * data['quality']

# Create a new feature combining square footage variables
data['combined_area'] = data['living_measure'] + data['lot_measure'] + data['total_area']

# Create house age and renovation age features
data['house_age'] = data['sale_year'] - data['yr_built']
data['renovation_age'] = data['sale_year'] - data['yr_renovated']
# Replace negative renovation_age with 0 where houses were never renovated
data['renovation_age'] = data['renovation_age'].apply(lambda x: 0 if x < 0 else x)

data = data.dropna()

from sklearn.model_selection import train_test_split

# Define target (price) and features (all other columns)
X = data.drop(['price','dayhours'], axis=1)
y = data['price']

# Split the data into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression - RMSE: {rmse}, MAE: {mae}, R²: {r2}')


from sklearn.ensemble import RandomForestRegressor

# Train a random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - RMSE: {rmse_rf}, MAE: {mae_rf}, R²: {r2_rf}')

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print(f'Best Parameters: {grid_search.best_params_}')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Calculate RMSE, MAE, and R² for the final model
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Final Model - RMSE: {rmse}, MAE: {mae}, R²: {r2}')


