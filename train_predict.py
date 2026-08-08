import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 3956 * c
    return miles

def preprocess_and_engineer(df, equip_medians=None, global_medians=None):
    """
    Applies consistent preprocessing and feature engineering to the data.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Coordinate Imputation (for December chart inputs if they don't exist)
    # Lexington: 36.99152, -84.99876
    # Fort Wayne: 41.31561, -85.36206
    if 'pickup_lat' not in df.columns:
        df['pickup_lat'] = np.where(df['pickup'] == 'Lexington', 36.99152, np.nan)
        df['pickup_lon'] = np.where(df['pickup'] == 'Lexington', -84.99876, np.nan)
    else:
        df['pickup_lat'] = df['pickup_lat'].fillna(36.99152)
        df['pickup_lon'] = df['pickup_lon'].fillna(-84.99876)
        
    if 'delivery_lat' not in df.columns:
        df['delivery_lat'] = np.where(df['delivery'] == 'Fort Wayne', 41.31561, np.nan)
        df['delivery_lon'] = np.where(df['delivery'] == 'Fort Wayne', -85.36206, np.nan)
    else:
        df['delivery_lat'] = df['delivery_lat'].fillna(41.31561)
        df['delivery_lon'] = df['delivery_lon'].fillna(-85.36206)

    # 2. Extract Date Features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['days_since_start'] = (df['date'] - pd.to_datetime('2025-01-01')).dt.days
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    # 3. Spatial Features
    df['haversine_distance'] = haversine_np(df['pickup_lon'], df['pickup_lat'], df['delivery_lon'], df['delivery_lat'])
    df['lat_diff'] = np.abs(df['pickup_lat'] - df['delivery_lat'])
    df['lon_diff'] = np.abs(df['pickup_lon'] - df['delivery_lon'])
    
    # 4. Impute Missing Weights and Market Indices
    df['weight'] = df['weight'].astype(float)
    if equip_medians is not None:
        for eq_type, med_val in equip_medians.items():
            df.loc[(df['equipment'] == eq_type) & (df['weight'].isnull()), 'weight'] = med_val
    if global_medians is not None:
        df['market_index'] = df['market_index'].fillna(global_medians['market_index'])
        df['quote_signal'] = df['quote_signal'].fillna(global_medians['quote_signal'])
        
    # 5. Convert Categorical Columns to category Dtype for LightGBM
    for col in ['pickup', 'delivery', 'equipment']:
        df[col] = df[col].astype('category')
        
    return df

def main():
    print("Loading datasets...")
    train_df = pd.read_csv('data/train-test.csv')
    val_df = pd.read_csv('data/validation.csv')
    dec_df = pd.read_csv('data/december-chart-inputs.csv')
    
    train_df['date'] = pd.to_datetime(train_df['date'])
    val_df['date'] = pd.to_datetime(val_df['date'])
    dec_df['date'] = pd.to_datetime(dec_df['date'])
    
    # --- Compute Medians from Training Set (To Avoid Leakage) ---
    equip_medians = train_df.groupby('equipment')['weight'].median().to_dict()
    global_medians = {
        'market_index': train_df['market_index'].median(),
        'quote_signal': train_df['quote_signal'].median()
    }
    
    # --- December Feature Preparation ---
    # Merge daily mean market_index and quote_signal from validation.csv to fill December missing features
    dec_daily_stats = val_df[val_df['date'].dt.month == 12].groupby('date')[['market_index', 'quote_signal']].mean().reset_index()
    dec_df = pd.merge(dec_df, dec_daily_stats, on='date', how='left')
    
    # --- Apply Preprocessing & Feature Engineering ---
    print("Preprocessing and engineering features...")
    train_feat = preprocess_and_engineer(train_df, equip_medians, global_medians)
    val_feat = preprocess_and_engineer(val_df, equip_medians, global_medians)
    dec_feat = preprocess_and_engineer(dec_df, equip_medians, global_medians)
    
    # Select Features for the Model
    features = [
        'pickup', 'delivery', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon',
        'distance', 'equipment', 'weight', 'market_index', 'quote_signal',
        'day_of_week', 'day_of_year', 'days_since_start', 'sin_day_of_year', 'cos_day_of_year',
        'haversine_distance', 'lat_diff', 'lon_diff'
    ]
    
    # --- 1. LOCAL TIME-BASED VALIDATION SPLIT ---
    # Train: Jan - Aug 2025 (before 2025-09-01)
    # Val: Sept - Oct 2025 (after 2025-09-01)
    print("\n--- Running Time-Based Local Validation ---")
    split_date = pd.to_datetime('2025-09-01')
    
    train_split = train_feat[train_feat['date'] < split_date]
    val_split = train_feat[train_feat['date'] >= split_date]
    
    X_train_split = train_split[features]
    y_train_split = np.log1p(train_split['posted_rate'])
    
    X_val_split = val_split[features]
    y_val_split_orig = val_split['posted_rate']
    
    # LightGBM Model Config
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 8,
        'min_data_in_leaf': 20,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    
    # Train validation model
    train_data = lgb.Dataset(X_train_split, label=y_train_split)
    val_model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=1200
    )
    
    # Evaluate Validation Model
    val_preds_log = val_model.predict(X_val_split)
    val_preds = np.expm1(val_preds_log)
    
    # Regression Metrics
    mae = mean_absolute_error(y_val_split_orig, val_preds)
    rmse = np.sqrt(mean_squared_error(y_val_split_orig, val_preds))
    r2 = r2_score(y_val_split_orig, val_preds)
    mape = np.mean(np.abs((y_val_split_orig - val_preds) / y_val_split_orig)) * 100
    
    print(f"Local Validation MAE:  ${mae:.2f}")
    print(f"Local Validation RMSE: ${rmse:.2f}")
    print(f"Local Validation MAPE: {mape:.2f}%")
    print(f"Local Validation R2:   {r2:.4f}")
    
    # --- 2. TRAIN FINAL MODEL ON FULL DATASET ---
    print("\nTraining final model on full dataset...")
    X_full = train_feat[features]
    y_full = np.log1p(train_feat['posted_rate'])
    
    full_train_data = lgb.Dataset(X_full, label=y_full)
    final_model = lgb.train(
        lgb_params,
        full_train_data,
        num_boost_round=1200
    )
    
    # --- 3. GENERATE SUBMISSION PREDICTIONS ---
    print("Generating predictions for validation set...")
    val_preds_log = final_model.predict(val_feat[features])
    val_preds_orig = np.expm1(val_preds_log)
    
    # Clip predictions to be strictly positive
    val_preds_orig = np.clip(val_preds_orig, 1.0, None)
    
    # Fill validation predictions file
    predictions_df = pd.DataFrame({
        'load_id': val_df['load_id'],
        'predicted_rate': val_preds_orig
    })
    
    # Check predictions format
    predictions_df.to_csv('validation_predictions.csv', index=False)
    print("Saved validation predictions to 'validation_predictions.csv'")
    
    # --- 4. GENERATE DECEMBER SCENARIO PREDICTIONS ---
    print("Generating predictions for December inputs...")
    dec_preds_log = final_model.predict(dec_feat[features])
    dec_preds_orig = np.expm1(dec_preds_log)
    
    # Clip predictions to be strictly positive
    dec_preds_orig = np.clip(dec_preds_orig, 1.0, None)
    
    dec_out = dec_df.copy()
    dec_out['predicted_rate'] = dec_preds_orig
    
    # We must save it to data/december-chart-inputs.csv containing the original columns plus filled predicted_rate
    dec_cols = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']
    
    # Format date back to string as expected by score.py
    dec_out['date'] = dec_out['date'].dt.strftime('%Y-%m-%d')
    dec_out[dec_cols].to_csv('data/december-chart-inputs.csv', index=False)
    print("Saved December inputs to 'data/december-chart-inputs.csv'")
    
if __name__ == '__main__':
    main()
