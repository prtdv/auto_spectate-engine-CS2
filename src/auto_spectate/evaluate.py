import os
import math
import numpy as np
import pandas as pd
import joblib
from demoparser2 import DemoParser
from collections import defaultdict
import bisect

TICK_RATE = 64.0
LOOKAHEAD_TICKS = 320

TEST_DEMO_PATH = r"demos\iem-cologne-major-2026-natus-vincere-vs-spirit-bo3-kgjfQml_20SbX4SdXD-5FD\natus-vincere-vs-spirit-m2-anubis.dem"

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

def clean_steamid(val):
    if pd.isna(val):
        return ""
    try:
        return str(int(float(val)))
    except:
        return str(val).strip()

def main():
    print(f"Loading model and evaluating on: {TEST_DEMO_PATH}...")
    if not os.path.exists("models/model.pkl"):
        print("Error: model.pkl not found. Please train the model first.")
        return
        
    model = joblib.load("models/model.pkl")
    
    # Load and process the test demo (just like in parse_demos.py)
    df_all = pd.read_csv("data/training_dataset.csv")
    test_samples_df = df_all[df_all["demo_name"] == os.path.basename(TEST_DEMO_PATH)]
    
    if test_samples_df.empty:
        print("Error: No test samples found in data/training_dataset.csv for this demo.")
        return
        
    print(f"Loaded {len(test_samples_df)} test samples from CSV.")
    
    # We need to re-predict and rank players at each timestamp
    # Let's group by timestamp (each timestamp is a tick)
    grouped = test_samples_df.groupby("timestamp")
    
    # Also parse the actual demo events to compute evaluation metrics
    parser = DemoParser(TEST_DEMO_PATH)
    player_hurt_df = parser.parse_event("player_hurt")
    player_death_df = parser.parse_event("player_death")
    
    # Clean steamids
    for df in [player_hurt_df, player_death_df]:
        for col in ["attacker_steamid", "user_steamid", "assister_steamid"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_steamid)
                
    # Build maps for combat events
    player_hurt_attacker = defaultdict(list)
    player_hurt_user = defaultdict(list)
    player_kills = defaultdict(list)
    
    # List of all kills in the match for Kill Coverage check
    all_kills_list = []
    
    if not player_hurt_df.empty:
        for _, row in player_hurt_df.iterrows():
            att = row.get("attacker_steamid", "")
            vic = row.get("user_steamid", "")
            t = int(row["tick"])
            dmg = int(row.get("dmg_health", 0))
            if att and att != vic:
                player_hurt_attacker[att].append(t)
            if vic:
                player_hurt_user[vic].append(t)
                
    if not player_death_df.empty:
        for _, row in player_death_df.iterrows():
            att = row.get("attacker_steamid", "")
            vic = row.get("user_steamid", "")
            t = int(row["tick"])
            if att and att != vic:
                player_kills[att].append(t)
                all_kills_list.append((t, att, vic))
                
    all_kills_list.sort(key=lambda x: x[0])
    
    # We also need a mapping from player names to steamids for the test samples
    # We can parse player info to map name -> steamid, or read it from parse_ticks.
    # To be fast, let's extract name to steamid map from a small parse_ticks call
    print("Mapping player names to steamids...")
    player_info = parser.parse_ticks(["name", "steamid"])
    name_to_sid = {}
    for _, row in player_info.dropna(subset=["name", "steamid"]).iterrows():
        name_to_sid[row["name"]] = clean_steamid(row["steamid"])
    print(f"Mapped {len(name_to_sid)} players.")

    # Metric variables
    total_windows = 0
    combat_active_windows = 0
    total_kill_windows = 0
    successful_kill_windows = 0
    lead_times = []
    
    print("\nRunning offline replay simulation and ranking...")
    for ts, group in grouped:
        tick = int(round(ts * TICK_RATE))
        
        # Predict scores for all alive players at this timestamp
        X = group[FEATURE_COLS]
        preds = model.predict(X)
        
        # Rank players
        rankings = []
        for idx, (_, row) in enumerate(group.iterrows()):
            p_name = row["player"]
            score = preds[idx]
            rankings.append((p_name, score))
            
        rankings.sort(key=lambda x: x[1], reverse=True)
        rec_player = rankings[0][0]
        rec_sid = name_to_sid.get(rec_player, "")
        
        if not rec_sid:
            continue
            
        total_windows += 1
        
        # Define lookahead window
        window_end_tick = tick + LOOKAHEAD_TICKS
        
        # 1. Kill Coverage
        # Check if there are any kills in the next 5 seconds
        kills_in_window = [k for k in all_kills_list if tick <= k[0] <= window_end_tick]
        if kills_in_window:
            total_kill_windows += 1
            # Check if the recommended player was the attacker in any of these kills
            rec_killed = any(k[1] == rec_sid for k in kills_in_window)
            if rec_killed:
                successful_kill_windows += 1
                
        # 2. Combat Coverage & Lead Time
        # Find all combat events for recommended player in next 5 seconds
        rec_hurt_att = player_hurt_attacker.get(rec_sid, [])
        rec_hurt_usr = player_hurt_user.get(rec_sid, [])
        rec_k = player_kills.get(rec_sid, [])
        
        combat_ticks = []
        # Add deals damage ticks
        idx_s = bisect.bisect_left(rec_hurt_att, tick)
        idx_e = bisect.bisect_right(rec_hurt_att, window_end_tick)
        combat_ticks.extend(rec_hurt_att[idx_s:idx_e])
        
        # Add receives damage ticks
        idx_s = bisect.bisect_left(rec_hurt_usr, tick)
        idx_e = bisect.bisect_right(rec_hurt_usr, window_end_tick)
        combat_ticks.extend(rec_hurt_usr[idx_s:idx_e])
        
        # Add kill ticks
        idx_s = bisect.bisect_left(rec_k, tick)
        idx_e = bisect.bisect_right(rec_k, window_end_tick)
        combat_ticks.extend(rec_k[idx_s:idx_e])
        
        if combat_ticks:
            combat_active_windows += 1
            first_combat_tick = min(combat_ticks)
            lead_time = (first_combat_tick - tick) / TICK_RATE
            lead_times.append(lead_time)
            
    # Calculate final metrics
    kill_coverage = (successful_kill_windows / total_kill_windows) if total_kill_windows > 0 else 0.0
    combat_coverage = (combat_active_windows / total_windows) if total_windows > 0 else 0.0
    avg_lead_time = np.mean(lead_times) if lead_times else 0.0
    
    print("\n--- Required Observer Metrics ---")
    print(f"Kill Coverage:   {kill_coverage * 100:.2f}% ({successful_kill_windows}/{total_kill_windows} windows)")
    print(f"Combat Coverage: {combat_coverage * 100:.2f}% ({combat_active_windows}/{total_windows} windows)")
    print(f"Avg Lead Time:   {avg_lead_time:.4f} seconds")

if __name__ == "__main__":
    main()
