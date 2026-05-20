import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from xgboost import XGBClassifier

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("personalised_dataset.csv")

# =========================
# 2. Tentukan fitur dan target
# =========================
cols_to_drop = [
    'Patient_ID',
    'Blood_Pressure',
    'Health_Risk',
    'Predicted_Insurance_Cost',
    'Diet_Recommendation',
    'Exercise_Recommendation',
    'Heart_Disease_Risk',
    'Diabetes_Risk'
]

X = df.drop(columns=cols_to_drop)

target_mapping = {
    'Low': 0,
    'Moderate': 1,
    'High': 2
}

y_heart = df['Heart_Disease_Risk'].map(target_mapping)
y_diabetes = df['Diabetes_Risk'].map(target_mapping)

y = pd.concat([y_heart, y_diabetes], axis=1)
y.columns = ['Heart_Disease_Risk', 'Diabetes_Risk']

# =========================
# 3. Preprocessing
# =========================
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# =========================
# 4. Split data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 5. Random Forest Model
# =========================
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    ))
])

rf_pipeline.fit(X_train, y_train)

# =========================
# 6. XGBoost Model
# =========================
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(
        XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='mlogloss'
        )
    ))
])

xgb_pipeline.fit(X_train, y_train)

# =========================
# 7. Simpan model
# =========================
with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf_pipeline, file)

with open("xgb_model.pkl", "wb") as file:
    pickle.dump(xgb_pipeline, file)

print("Model berhasil dilatih dan disimpan.")