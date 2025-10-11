import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Custom transformer to create 'Damage_Potential' feature
class DamagePotentialCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        mag = X['Magnitude']
        depth = X['Depth']
        damage_potential = 0.6 * mag + 0.2 * (700 - depth) / 700 * 10
        X = X.copy()
        X['Damage_Potential'] = damage_potential
        return X

# Select relevant features including categorical and for imputation
def select_features(X):
    return X[['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Type', 'Magnitude Type', 'Status', 'Root Mean Square']]


categorical_cols = ['Magnitude Type', 'Status']
numerical_cols = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Root Mean Square']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# Impute missing 'Root Mean Square' using RandomForestRegressor
def impute_root_mean_square(df):
    known_df = df[df['Root Mean Square'].notna()]
    unknown_df = df[df['Root Mean Square'].isna()]

    if unknown_df.empty:
        return df

    features = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
    X_train = known_df[features]
    y_train = known_df['Root Mean Square']
    X_pred = unknown_df[features]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predicted = model.predict(X_pred)
    df.loc[df['Root Mean Square'].isna(), 'Root Mean Square'] = predicted
    return df

# Named functions replacing lambdas
def filter_earthquake(X):
    return X[X['Type'] == 'Earthquake']

def drop_type(X):
    return X.drop('Type', axis=1)

# Pipeline for categorical encoding (custom one-hot encoding)
def encode_categoricals(X):
    categorical_cols = ['Magnitude Type', 'Status']
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=X.index)
    X = X.drop(columns=categorical_cols).reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)
    return pd.concat([X, encoded_df], axis=1)

# Complete pipeline creation
def create_preprocessing_pipeline():
    return Pipeline(steps=[
        ('select_features', FunctionTransformer(select_features, validate=False)),
        ('impute_rms', FunctionTransformer(impute_root_mean_square, validate=False)),
        ('filter_earthquake', FunctionTransformer(filter_earthquake, validate=False)),
        ('drop_type', FunctionTransformer(drop_type, validate=False)),
        ('damage_potential', DamagePotentialCreator()),
        ('encode_categoricals', FunctionTransformer(encode_categoricals, validate=False)),
    ])

pipeline = create_preprocessing_pipeline()

# # Save pipeline
# joblib.dump(pipeline, 'models/data_preprocessing_pipeline.pkl')
