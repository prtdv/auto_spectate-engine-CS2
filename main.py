import os
import json
import pandas as pd
from src.auto_spectate.api import SpectatorRecommender

def main():
    print("====================================================")
    print("CS2 Auto-Spectator Recommendation System (MVP)")
    print("====================================================\n")
    
    # Check if dataset and model files exist
    dataset_exists = os.path.exists("data/training_dataset.csv")
    model_exists = os.path.exists("models/model.pkl")
    
    if not dataset_exists:
        print("Step 1: Running Demo Parsing Pipeline to generate dataset...")
        from src.auto_spectate import parse_demos
        parse_demos.main()
    else:
        print("Step 1: Dataset 'data/training_dataset.csv' already exists.")
        
    if not model_exists:
        print("\nStep 2: Training the RandomForestRegressor model...")
        from src.auto_spectate import train
        train.main()
    else:
        print("\nStep 2: Trained model 'models/model.pkl' already exists.")
        
    print("\nStep 3: Running offline evaluation on unseen test demo...")
    from src.auto_spectate import evaluate
    evaluate.main()
    
    print("\nStep 4: Demonstrating live prediction API...")
    # Instantiate API
    recommender = SpectatorRecommender(model_path="models/model.pkl")
    
    # Create sample live input representing 5 alive players in a match state
    sample_live_players = [
        {
            "player": "Aleksib",
            "hp": 85,
            "armor": 100,
            "weapon_tier": 3,  # Rifle
            "speed": 220.0,
            "nearest_enemy_distance": 450.0,
            "enemy_count_500": 1,
            "enemy_count_1000": 2,
            "damage_dealt_last_5s": 45.0,
            "damage_taken_last_5s": 15.0,
            "shots_fired_last_5s": 4,
            "utility_thrown_last_5s": 0,
            "kills_last_30s": 1,
            "time_since_last_combat": 1.2,
            "view_angle_to_enemy": 0.85,
            "is_bomb_planted": 0,
            "is_scoped": 0
        },
        {
            "player": "ZywOo",
            "hp": 100,
            "armor": 100,
            "weapon_tier": 4,  # AWP
            "speed": 110.0,
            "nearest_enemy_distance": 950.0,
            "enemy_count_500": 0,
            "enemy_count_1000": 1,
            "damage_dealt_last_5s": 0.0,
            "damage_taken_last_5s": 0.0,
            "shots_fired_last_5s": 0,
            "utility_thrown_last_5s": 0,
            "kills_last_30s": 2,
            "time_since_last_combat": 8.5,
            "view_angle_to_enemy": 0.95,
            "is_bomb_planted": 0,
            "is_scoped": 1
        },
        {
            "player": "w0nderful",
            "hp": 20,
            "armor": 50,
            "weapon_tier": 1,  # Pistol
            "speed": 250.0,
            "nearest_enemy_distance": 250.0,
            "enemy_count_500": 1,
            "enemy_count_1000": 1,
            "damage_dealt_last_5s": 0.0,
            "damage_taken_last_5s": 80.0,
            "shots_fired_last_5s": 1,
            "utility_thrown_last_5s": 0,
            "kills_last_30s": 0,
            "time_since_last_combat": 0.2,
            "view_angle_to_enemy": -0.4,
            "is_bomb_planted": 0,
            "is_scoped": 0
        },
        {
            "player": "flameZ",
            "hp": 100,
            "armor": 100,
            "weapon_tier": 3,  # Rifle
            "speed": 0.0,  # Holding angle
            "nearest_enemy_distance": 1200.0,
            "enemy_count_500": 0,
            "enemy_count_1000": 0,
            "damage_dealt_last_5s": 0.0,
            "damage_taken_last_5s": 0.0,
            "shots_fired_last_5s": 0,
            "utility_thrown_last_5s": 1,
            "kills_last_30s": 0,
            "time_since_last_combat": 25.0,
            "view_angle_to_enemy": 0.1,
            "is_bomb_planted": 0,
            "is_scoped": 0
        },
        {
            "player": "b1t",
            "hp": 90,
            "armor": 100,
            "weapon_tier": 3,  # Rifle
            "speed": 210.0,
            "nearest_enemy_distance": 310.0,
            "enemy_count_500": 1,
            "enemy_count_1000": 3,
            "damage_dealt_last_5s": 88.0,
            "damage_taken_last_5s": 0.0,
            "shots_fired_last_5s": 6,
            "utility_thrown_last_5s": 0,
            "kills_last_30s": 2,
            "time_since_last_combat": 0.5,
            "view_angle_to_enemy": 0.98,
            "is_bomb_planted": 0,
            "is_scoped": 0
        }
    ]
    
    result = recommender.recommend(sample_live_players)
    
    print("\nAPI Response Output:")
    print(json.dumps(result, indent=4))
    
    print("\nFinal Recommended spectator target:")
    print(f"-> SPECTATE: {result['recommended_player']} (predicted score: {result['predicted_score']})")

if __name__ == "__main__":
    main()
