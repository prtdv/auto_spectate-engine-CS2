import os
import joblib
import pandas as pd

class SpectatorRecommender:
    def __init__(self, model_path="models/model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        self.model = joblib.load(model_path)
        self.feature_cols = [
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

    def recommend(self, player_features_list):
        """
        Predict future combat scores and recommend the best player to spectate.
        
        Input:
            player_features_list (list of dicts): list containing feature dictionaries for each alive player:
                [
                    {
                        "player": "PlayerName",
                        "hp": 100,
                        "armor": 100,
                        "weapon_tier": 3,
                        "speed": 250.0,
                        "nearest_enemy_distance": 320.0,
                        "enemy_count_500": 1,
                        "enemy_count_1000": 2,
                        "damage_dealt_last_5s": 50,
                        "damage_taken_last_5s": 0,
                        "shots_fired_last_5s": 3,
                        "kills_last_30s": 1,
                        "time_since_last_combat": 2.5
                    },
                    ...
                ]
                
        Output:
            dict: {
                "recommended_player": "PlayerName",
                "predicted_score": 210.4,
                "rankings": [
                    { "player": "PlayerName", "score": 210.4 },
                    { "player": "PlayerName2", "score": 180.2 }
                ]
            }
        """
        if not player_features_list:
            return {
                "recommended_player": None,
                "predicted_score": 0.0,
                "rankings": []
            }

        # Convert to pandas DataFrame for model prediction
        df = pd.DataFrame(player_features_list)
        
        # Verify columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column: {col}")
                
        # Extract features and predict
        X = df[self.feature_cols]
        scores = self.model.predict(X)
        
        # Assemble rankings
        rankings = []
        for i, row in df.iterrows():
            rankings.append({
                "player": row["player"],
                "score": round(float(scores[i]), 2)
            })
            
        # Sort rankings from highest score to lowest score
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "recommended_player": rankings[0]["player"],
            "predicted_score": rankings[0]["score"],
            "rankings": rankings
        }
