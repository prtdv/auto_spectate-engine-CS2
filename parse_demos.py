import os
import math
import numpy as np
import pandas as pd
from demoparser2 import DemoParser
from collections import defaultdict
import bisect

# Constant definition
TICK_RATE = 64.0  # 64 ticks per second
LOOKAHEAD_TICKS = 320  # 5 seconds
LOOKBACK_5S_TICKS = 320  # 5 seconds
LOOKBACK_30S_TICKS = 1920  # 30 seconds

# Demos paths
DEMO_DIRS = [
    r"demos\iem-atlanta-2026-natus-vincere-vs-vitality-bo3-Oy5NkCJPWmHl6H0cKtC9_L",
    r"demos\iem-cologne-major-2026-natus-vincere-vs-spirit-bo3-kgjfQml_20SbX4SdXD-5FD"
]

def get_weapon_tier(weapon_name):
    if not weapon_name or pd.isna(weapon_name):
        return 1
    w_lower = str(weapon_name).lower()
    # Snipers (Tier 4)
    if any(x in w_lower for x in ["awp", "scar-20", "scar20", "g3sg1"]):
        return 4
    # Rifles (Tier 3)
    elif any(x in w_lower for x in ["ak-47", "ak47", "m4a4", "m4a1-s", "m4a1", "galil", "famas", "sg 553", "sg553", "sg556", "aug", "ssg 08", "ssg08"]):
        return 3
    # SMGs / Heavy (Tier 2)
    elif any(x in w_lower for x in ["mac-10", "mac10", "mp9", "mp7", "mp5", "ump", "p90", "bizon", "mag-7", "mag7", "xm1014", "nova", "sawed-off", "sawedoff", "negev", "m249"]):
        return 2
    # Pistols (Tier 1)
    elif any(x in w_lower for x in ["usp", "glock", "p250", "five-seven", "fiveseven", "tec-9", "tec9", "deagle", "desert eagle", "elite", "cz75", "hkp2000", "revolver"]):
        return 1
    else:
        return 1  # Default

def clean_steamid(val):
    if pd.isna(val):
        return ""
    try:
        # handle float representation of steamid
        return str(int(float(val)))
    except:
        return str(val).strip()

def process_demo(demo_path):
    demo_name = os.path.basename(demo_path)
    print(f"Processing demo: {demo_name}...")
    parser = DemoParser(demo_path)
    
    # 1. Parse round events to find live round intervals
    freeze_ends = parser.parse_event("round_freeze_end")
    ends = parser.parse_event("round_end")
    
    if freeze_ends.empty or ends.empty:
        print(f"Skipping {demo_name} due to missing round events.")
        return []
        
    freeze_ticks = sorted(freeze_ends["tick"].unique())
    end_ticks = sorted(ends["tick"].unique())
    
    # Pair freeze_end and round_end chronologically
    live_rounds = []
    round_counter = 1
    for et in end_ticks:
        # Find the closest freeze_end before this round_end
        prior_freezes = [ft for ft in freeze_ticks if ft < et]
        if not prior_freezes:
            continue
        ft = prior_freezes[-1]
        # Check if there is another round_end between ft and et
        other_ends = [oet for oet in end_ticks if ft < oet < et]
        if not other_ends:
            live_rounds.append((round_counter, ft, et))
            round_counter += 1
            
    print(f"Found {len(live_rounds)} valid live rounds.")
    
    # Determine the ticks we will sample
    sample_ticks = []
    tick_to_round_and_num = {}
    for r_num, ft, et in live_rounds:
        # Sample once every second (64 ticks), starting at freeze_end
        # Discard samples with less than 5 seconds (320 ticks) remaining in the round
        r_ticks = list(range(ft, et - LOOKAHEAD_TICKS + 1, 64))
        sample_ticks.extend(r_ticks)
        for t in r_ticks:
            tick_to_round_and_num[t] = (r_num, et)
            
    if not sample_ticks:
        print(f"No sample ticks generated for {demo_name}.")
        return []
        
    print(f"Generated {len(sample_ticks)} sample ticks.")
    
    # 2. Parse event dataframes for rolling features and labels
    player_hurt_df = parser.parse_event("player_hurt")
    player_death_df = parser.parse_event("player_death")
    weapon_fire_df = parser.parse_event("weapon_fire")
    
    # Convert steamids
    for df in [player_hurt_df, player_death_df, weapon_fire_df]:
        for col in ["attacker_steamid", "user_steamid", "assister_steamid"]:
            if col in df.columns:
                df[col] = df[col].apply(clean_steamid)
                
    # Build fast lookups
    player_hurt_attacker = defaultdict(list)
    player_hurt_user = defaultdict(list)
    player_weapon_fire = defaultdict(list)
    player_kills = defaultdict(list)
    player_assists = defaultdict(list)
    
    if not player_hurt_df.empty:
        for _, row in player_hurt_df.iterrows():
            att = row.get("attacker_steamid", "")
            vic = row.get("user_steamid", "")
            t = int(row["tick"])
            dmg = int(row.get("dmg_health", 0))
            if att and att != vic:
                player_hurt_attacker[att].append((t, dmg))
            if vic:
                player_hurt_user[vic].append((t, dmg))
                
    if not player_death_df.empty:
        for _, row in player_death_df.iterrows():
            att = row.get("attacker_steamid", "")
            vic = row.get("user_steamid", "")
            ast = row.get("assister_steamid", "")
            t = int(row["tick"])
            if att and att != vic:
                player_kills[att].append(t)
            if ast:
                player_assists[ast].append(t)
                
    if not weapon_fire_df.empty:
        for _, row in weapon_fire_df.iterrows():
            user = row.get("user_steamid", "")
            t = int(row["tick"])
            if user:
                player_weapon_fire[user].append(t)
                
    # Sort lists
    for k in player_hurt_attacker: player_hurt_attacker[k].sort(key=lambda x: x[0])
    for k in player_hurt_user: player_hurt_user[k].sort(key=lambda x: x[0])
    for k in player_weapon_fire: player_weapon_fire[k].sort()
    for k in player_kills: player_kills[k].sort()
    for k in player_assists: player_assists[k].sort()
    
    # Helper range sum/count functions
    def sum_hurt_in_range(hurt_list, start_t, end_t):
        if not hurt_list: return 0
        ticks = [x[0] for x in hurt_list]
        idx_start = bisect.bisect_left(ticks, start_t)
        idx_end = bisect.bisect_right(ticks, end_t)
        return sum(hurt_list[i][1] for i in range(idx_start, idx_end))
        
    def count_in_range(tick_list, start_t, end_t):
        if not tick_list: return 0
        idx_start = bisect.bisect_left(tick_list, start_t)
        idx_end = bisect.bisect_right(tick_list, end_t)
        return idx_end - idx_start
        
    def get_time_since_last_combat(steamid, current_t):
        last_att = -1
        att_list = player_hurt_attacker.get(steamid, [])
        if att_list:
            ticks = [x[0] for x in att_list]
            idx = bisect.bisect_right(ticks, current_t) - 1
            if idx >= 0: last_att = ticks[idx]
            
        last_vic = -1
        vic_list = player_hurt_user.get(steamid, [])
        if vic_list:
            ticks = [x[0] for x in vic_list]
            idx = bisect.bisect_right(ticks, current_t) - 1
            if idx >= 0: last_vic = ticks[idx]
            
        last_t = max(last_att, last_vic)
        if last_t == -1: return 99.0
        return (current_t - last_t) / 64.0

    # 3. Parse player ticks data
    tick_props = [
        "health", "armor_value", "X", "Y", "Z", "yaw", "pitch",
        "velocity_X", "velocity_Y", "velocity_Z", "active_weapon_name",
        "team_num", "steamid", "name"
    ]
    
    ticks_df = parser.parse_ticks(tick_props)
    ticks_df = ticks_df[ticks_df["tick"].isin(sample_ticks)]
    
    # Group by tick to build samples
    grouped = ticks_df.groupby("tick")
    samples = []
    
    for tick, group in grouped:
        r_num, et = tick_to_round_and_num[tick]
        
        # Filter for alive players on CT or T
        alive_players = group[(group["health"] > 0) & (group["team_num"].isin([2, 3]))].copy()
        if alive_players.empty:
            continue
            
        alive_players["clean_sid"] = alive_players["steamid"].apply(clean_steamid)
        
        # Extract lists for distance calculations
        players_data = alive_players[["name", "clean_sid", "team_num", "X", "Y", "Z"]].values
        
        # Map players to coordinates and calculate distances
        coords = alive_players[["X", "Y", "Z"]].values.astype(float)
        # Compute pairwise distance matrix
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Precompute features per player
        for idx, row in alive_players.reset_index(drop=True).iterrows():
            p_name = row["name"]
            sid = row["clean_sid"]
            team = row["team_num"]
            hp = int(row["health"])
            armor = int(row["armor_value"])
            w_tier = get_weapon_tier(row["active_weapon_name"])
            
            vx = float(row["velocity_X"])
            vy = float(row["velocity_Y"])
            speed = math.sqrt(vx**2 + vy**2)
            
            # Find distances to alive enemies
            enemy_dists = []
            for j, p_other in enumerate(players_data):
                if p_other[2] != team:  # different team
                    enemy_dists.append(dists[idx, j])
                    
            if not enemy_dists:
                nearest_enemy_distance = 99999.0
                enemy_count_500 = 0
                enemy_count_1000 = 0
            else:
                nearest_enemy_distance = min(enemy_dists)
                enemy_count_500 = sum(d <= 500 for d in enemy_dists)
                enemy_count_1000 = sum(d <= 1000 for d in enemy_dists)
                
            # Rolling window features
            damage_dealt_last_5s = sum_hurt_in_range(player_hurt_attacker[sid], tick - LOOKBACK_5S_TICKS, tick)
            damage_taken_last_5s = sum_hurt_in_range(player_hurt_user[sid], tick - LOOKBACK_5S_TICKS, tick)
            shots_fired_last_5s = count_in_range(player_weapon_fire[sid], tick - LOOKBACK_5S_TICKS, tick)
            kills_last_30s = count_in_range(player_kills[sid], tick - LOOKBACK_30S_TICKS, tick)
            time_since_last_combat = get_time_since_last_combat(sid, tick)
            
            # Label calculations (future window [tick, tick + LOOKAHEAD_TICKS])
            fut_dmg_dealt = sum_hurt_in_range(player_hurt_attacker[sid], tick, tick + LOOKAHEAD_TICKS)
            fut_dmg_taken = sum_hurt_in_range(player_hurt_user[sid], tick, tick + LOOKAHEAD_TICKS)
            fut_assists = count_in_range(player_assists[sid], tick, tick + LOOKAHEAD_TICKS)
            fut_kills = count_in_range(player_kills[sid], tick, tick + LOOKAHEAD_TICKS)
            
            future_score = (
                1.0 * fut_dmg_dealt
                + 0.5 * fut_dmg_taken
                + 50.0 * fut_assists
                + 100.0 * fut_kills
            )
            
            samples.append({
                "demo_name": demo_name,
                "timestamp": tick / TICK_RATE,
                "round_number": r_num,
                "player": p_name,
                "hp": hp,
                "armor": armor,
                "weapon_tier": w_tier,
                "speed": speed,
                "nearest_enemy_distance": nearest_enemy_distance,
                "enemy_count_500": enemy_count_500,
                "enemy_count_1000": enemy_count_1000,
                "damage_dealt_last_5s": damage_dealt_last_5s,
                "damage_taken_last_5s": damage_taken_last_5s,
                "shots_fired_last_5s": shots_fired_last_5s,
                "kills_last_30s": kills_last_30s,
                "time_since_last_combat": time_since_last_combat,
                "future_score": future_score
            })
            
    print(f"Extracted {len(samples)} player-tick samples from {demo_name}.\n")
    return samples

def main():
    all_samples = []
    
    # Traverse through all demo directories and parse files
    for directory in DEMO_DIRS:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist. Skipping.")
            continue
        for file in os.listdir(directory):
            if file.endswith(".dem"):
                demo_path = os.path.join(directory, file)
                try:
                    all_samples.extend(process_demo(demo_path))
                except Exception as e:
                    print(f"Error processing demo {file}: {e}")
                    
    if not all_samples:
        print("No samples generated at all.")
        return
        
    df = pd.DataFrame(all_samples)
    output_csv = "training_dataset.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved dataset to {output_csv}. Total rows: {len(df)}")

if __name__ == "__main__":
    main()
