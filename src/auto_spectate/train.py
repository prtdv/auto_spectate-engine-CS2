import os
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define columns
FEATURE_COLS = [
    "hp",
    "armor",
    "weapon_tier",
    "speed",
    "nearest_enemy_distance",
    "enemy_count_500",
    "enemy_count_1000",
    "damage_dealt_last_5s",
    "damage_taken_last_5s",
    "shots_fired_last_5s",
    "utility_thrown_last_5s",
    "kills_last_30s",
    "time_since_last_combat",
    "view_angle_to_enemy",
    "is_bomb_planted",
    "is_scoped"
]
TARGET_COL = "future_score"
TEST_DEMO = "natus-vincere-vs-spirit-m2-anubis.dem"

def main():
    print("Loading dataset...")
    df = pd.read_csv("data/training_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Train/Test Split by Demo to prevent leakage
    train_df = df[df["demo_name"] != TEST_DEMO]
    test_df = df[df["demo_name"] == TEST_DEMO]
    
    print(f"Train samples (from {train_df['demo_name'].nunique()} demos): {len(train_df)}")
    print(f"Test samples (from {test_df['demo_name'].nunique()} demo - {TEST_DEMO}): {len(test_df)}")
    
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    
    print("Training LGBMRegressor model...")
    model = LGBMRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Training completed.")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Metrics (Test Set) ---")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # Feature Importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print("\n--- Feature Importances ---")
    print(feat_imp.to_string(index=False))
    
    # Save model
    model_path = "models/model.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to {model_path}")

if __name__ == "__main__":
    main()
