import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Generate synthetic EV charging data (in a real scenario, you'd load your dataset)
def generate_ev_data(num_samples=10000):
    np.random.seed(42)
    
    # Create time features
    timestamps = pd.date_range('2023-01-01', periods=num_samples, freq='H')
    hours = timestamps.hour
    days = timestamps.dayofweek
    months = timestamps.month
    
    # Create weather-like features
    temperature = 10 + 15 * np.sin(hours/24 * 2*np.pi) + np.random.normal(0, 3, num_samples)
    precipitation = np.random.poisson(0.1, num_samples)
    
    # Create location features
    locations = np.random.choice(['Residential', 'Commercial', 'Public', 'Workplace'], num_samples)
    
    # Create target variable - charging demand in kWh
    base_demand = 5 + 10 * np.sin((hours-8)/24 * 2*np.pi)  # Higher during daytime
    weekend_effect = np.where(days >= 5, 0.7, 1)  # Lower on weekends
    weather_effect = np.where(precipitation > 0, 0.8, 1)  # Lower when raining
    charging_demand = base_demand * weekend_effect * weather_effect * np.random.lognormal(0, 0.1, num_samples)
    charging_demand = np.round(np.clip(charging_demand, 0, 30), 2)
    
    # Create dataframe
    data = pd.DataFrame({
        'timestamp': timestamps,
        'hour': hours,
        'day_of_week': days,
        'month': months,
        'temperature': temperature,
        'precipitation': precipitation,
        'location_type': locations,
        'charging_demand_kWh': charging_demand
    })
    
    return data

# Load or generate data
ev_data = generate_ev_data()

# Feature engineering
ev_data['is_weekend'] = (ev_data['day_of_week'] >= 5).astype(int)
ev_data['time_of_day'] = pd.cut(ev_data['hour'], 
                               bins=[0, 6, 12, 18, 24],
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                               right=False)

# Define features and target
X = ev_data[['hour', 'day_of_week', 'month', 'temperature', 
             'precipitation', 'location_type', 'is_weekend', 'time_of_day']]
y = ev_data['charging_demand_kWh']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['hour', 'day_of_week', 'month', 'temperature', 'precipitation', 'is_weekend']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['location_type', 'time_of_day']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create and train model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Model Performance Metrics:")
print(f"Mean Absolute Error: {mae:.2f} kWh")
print(f"Mean Squared Error: {mse:.2f} kWh²")
print(f"Root Mean Squared Error: {rmse:.2f} kWh")

# Feature importance
rf_model = model.named_steps['regressor']
feature_names = numeric_features + list(model.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(categorical_features))

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importances:")
for i in indices:
    print(f"{feature_names[i]:<30} {importances[i]:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Charging Demand (kWh)')
plt.ylabel('Predicted Charging Demand (kWh)')
plt.title('Actual vs Predicted EV Charging Demand')
plt.show()

# Time series plot for a specific location
sample_data = X_test.copy()
sample_data['actual'] = y_test
sample_data['predicted'] = y_pred
sample_data = sample_data.sort_values(['month', 'day_of_week', 'hour'])

plt.figure(figsize=(14, 6))
plt.plot(sample_data['actual'].values[:100], label='Actual')
plt.plot(sample_data['predicted'].values[:100], label='Predicted')
plt.xlabel('Time (hours)')
plt.ylabel('Charging Demand (kWh)')
plt.title('EV Charging Demand Prediction Over Time')
plt.legend()
plt.show()
