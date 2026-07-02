import os
import json
from api import SpectatorRecommender

def build_player(name, hp=100, weapon_tier=3, speed=250.0, dist=1000.0, last_combat=99.0, deals_dmg=0.0):
    # Apply reasonable defaults for the remaining rolling window features
    return {
        "player": name,
        "hp": hp,
        "armor": 100 if hp > 0 else 0,
        "weapon_tier": weapon_tier,
        "speed": speed,
        "nearest_enemy_distance": dist,
        "enemy_count_500": 1 if dist <= 500 else 0,
        "enemy_count_1000": 1 if dist <= 1000 else 0,
        "damage_dealt_last_5s": deals_dmg,
        "damage_taken_last_5s": 0.0,
        "shots_fired_last_5s": 4 if deals_dmg > 0 else 0,
        "kills_last_30s": 1 if deals_dmg > 100 else 0,
        "time_since_last_combat": last_combat
    }

def run_scenario(recommender, players, scenario_title, description):
    print("\n" + "=" * 60)
    print(f"Scenario: {scenario_title}")
    print(f"Description: {description}")
    print("=" * 60)
    
    # Run recommendation
    result = recommender.recommend(players)
    
    # Print results nicely
    print("\nPlayer inputs evaluated:")
    for p in players:
        print(f" - {p['player']}: HP={p['hp']}, WeaponTier={p['weapon_tier']}, Speed={p['speed']:.1f}, EnemyDist={p['nearest_enemy_distance']:.1f}, LastCombat={p['time_since_last_combat']}s, RecentDmgDealt={p['damage_dealt_last_5s']}")
        
    print("\nAPI Output rankings:")
    for idx, r in enumerate(result["rankings"]):
        rec_marker = "⭐ RECOMMENDED" if r["player"] == result["recommended_player"] else ""
        print(f"  {idx + 1}. {r['player']:<12} | Score: {r['score']:>6.2f} {rec_marker}")
    print("=" * 60 + "\n")

def main():
    if not os.path.exists("model.pkl"):
        print("Error: model.pkl not found! Please run 'uv run python main.py' first to train the model.")
        return

    recommender = SpectatorRecommender(model_path="model.pkl")
    
    while True:
        print("\n--- CS2 Auto-Spectator Recommendation Tester ---")
        print("1. Scenario: Sniper holding angle vs Entry fragger rushing site")
        print("2. Scenario: Low-health player running away vs Player entering combat")
        print("3. Scenario: Quiet buy period (no combat yet) vs Active firefight")
        print("4. Custom: Create your own match state")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            players = [
                build_player("ZywOo (AWP)", hp=100, weapon_tier=4, speed=0.0, dist=1500.0, last_combat=45.0),
                build_player("apEX (Rushing)", hp=100, weapon_tier=3, speed=250.0, dist=350.0, last_combat=4.0, deals_dmg=50.0),
                build_player("flameZ (Passive)", hp=100, weapon_tier=3, speed=110.0, dist=2000.0, last_combat=99.0)
            ]
            run_scenario(
                recommender, players, 
                "AWP vs Entry Fragger", 
                "ZywOo is holding a far angle with an AWP (Tier 4, speed=0, dist=1500). apEX is rushing an entry path with a rifle (Tier 3, speed=250, dist=350, recently dealt 50 dmg)."
            )
            
        elif choice == "2":
            players = [
                build_player("w0nderful (Low HP)", hp=15, weapon_tier=1, speed=250.0, dist=800.0, last_combat=1.0),
                build_player("b1t (Engaged)", hp=90, weapon_tier=3, speed=150.0, dist=200.0, last_combat=0.5, deals_dmg=80.0),
                build_player("mezii (Passive)", hp=100, weapon_tier=3, speed=0.0, dist=1200.0, last_combat=99.0)
            ]
            run_scenario(
                recommender, players, 
                "Low HP Escape vs Active Engagement", 
                "w0nderful is very low health (15 HP) running away with a pistol. b1t is high health (90 HP) actively trading damage (dist=200, dealt 80 dmg half a second ago)."
            )
            
        elif choice == "3":
            players = [
                build_player("ropz (Lurking)", hp=100, weapon_tier=3, speed=110.0, dist=900.0, last_combat=99.0),
                build_player("iM (Firefight)", hp=75, weapon_tier=3, speed=220.0, dist=150.0, last_combat=0.2, deals_dmg=120.0),
                build_player("Aleksib (Holding)", hp=100, weapon_tier=3, speed=0.0, dist=1600.0, last_combat=99.0)
            ]
            run_scenario(
                recommender, players, 
                "Quiet vs Active Firefight", 
                "ropz is quietly lurking (no combat yet, dist=900). iM is in an active firefight, dealing 120 damage within the last 5s and 0.2s since last combat."
            )
            
        elif choice == "4":
            print("\n--- Create Custom Match State ---")
            num_players = int(input("How many players are alive? (2-5): ").strip())
            players = []
            for i in range(num_players):
                print(f"\n--- Player {i+1} ---")
                name = input("Player Name: ").strip()
                hp = int(input("HP (0-100): ").strip())
                tier = int(input("Weapon Tier (1=Pistol, 2=SMG, 3=Rifle, 4=AWP): ").strip())
                speed = float(input("Speed (0 = stationary, 250 = full run): ").strip())
                dist = float(input("Distance to Nearest Enemy (units): ").strip())
                last_c = float(input("Seconds since last combat (e.g. 0.5, or 99 if none): ").strip())
                recent_dmg = float(input("Damage dealt in last 5 seconds: ").strip())
                
                players.append(build_player(name, hp, tier, speed, dist, last_c, recent_dmg))
                
            run_scenario(recommender, players, "Custom Scenario", "Your custom interactive match state features.")
            
        elif choice == "5":
            print("Exiting interactive tester. Goodbye!")
            break
        else:
            print("Invalid choice, please enter 1-5.")

if __name__ == "__main__":
    main()
