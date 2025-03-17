import pandas as pd
import numpy as np
from random import choices
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pickle


# Load Dataset
file_path = 'galton_height_dataset.csv'
data = pd.read_csv(file_path)

# Add New Columns
# Add birth_order: Generate sequential birth order up to the 'kids' count within a family
birth_orders = []
for _, group in data.groupby('family'):
    birth_orders.extend(range(1, len(group) + 1))

data['birth_order'] = birth_orders

# Add diet_quality, play_sports, living_environment, grand_parent_height
data['diet_quality'] = choices(['high', 'medium', 'low'], k=len(data))
data['play_sports'] = choices(['yes', 'no'], k=len(data))
data['living_environment'] = choices(['urban', 'rural'], k=len(data))
data['grand_parent_height'] = data['father'] - np.random.randint(2, 8, size=len(data))

# Encode Categorical Variables
categorical_cols = ['diet_quality', 'play_sports', 'living_environment']
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Combine male and female into a single column named gender
# Assuming 'male' and 'female' columns already exist
data['gender'] = data['male']  # Male is 1, female is 0
data.drop(columns=['male', 'female'], inplace=True)

# Define Features and Target
target = 'height'
features = ['father', 'mother', 'grand_parent_height', 'birth_order', 'living_environment', 'diet_quality', 'play_sports', 'gender']

X = data[features]
y = data[target]

# Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM Model
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)
pickle.dump(lgb_model, open("model.pkl", "wb"))
print("Model training complete and saved as 'model.pkl'.")
