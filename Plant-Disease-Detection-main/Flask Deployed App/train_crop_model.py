import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('crop_suggestions.csv')

# Encode categorical variables
le_region = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df['Region_encoded'] = le_region.fit_transform(df['Region'])
df['Season_encoded'] = le_season.fit_transform(df['Season'])
df['Crop_encoded'] = le_crop.fit_transform(df['Crop'])

# Features and target
X = df[['Region_encoded', 'Season_encoded']]
y = df['Crop_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'crop_model.pkl')
joblib.dump(le_region, 'le_region.pkl')
joblib.dump(le_season, 'le_season.pkl')
joblib.dump(le_crop, 'le_crop.pkl')

print("Model trained and saved.")