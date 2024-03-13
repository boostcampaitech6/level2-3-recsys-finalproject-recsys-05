def process_summoner_data(df):
    # unlist
    df = df.explode(["champ_id", "champ_play", "champ_win", "champ_lose", 
                        "champ_kill", "champ_death", "champ_assist", "gold_earned", 
                        "minion_kill", "game_length_second", "vision_wards_bought_in_game", "vision_score",
                        "wards_placed", "wards_killed", "heal", "time_ccing_others", 
                        "damage_dealt_to_champions"]).reset_index(drop=True) # df 탐색후 첫 번째 열을 for문으로 돌려 타입 검사 -> 리스트일 경우 explode, obj일 경우 str

    # 데이터 타입 변경
    df["summoner_id"] = df["summoner_id"].astype(str)
    df["game_name"] = df["game_name"].astype(str)
    df["tagline"] = df["tagline"].astype(str)
    df["tier"] = df["tier"].astype(str)
    
    df["division"] = df["division"].astype(int)
    df["champ_id"] = df["champ_id"].astype(int)
    df["champ_play"] = df["champ_play"].astype(int)
    df["champ_win"] = df["champ_win"].astype(int)
    df["champ_lose"] = df["champ_lose"].astype(int)
    df["champ_kill"] = df["champ_kill"].astype(int)
    df["champ_death"] = df["champ_death"].astype(int)
    df["champ_assist"] = df["champ_assist"].astype(int)
    df["gold_earned"] = df["gold_earned"].astype(int)
    df["minion_kill"] = df["minion_kill"].astype(int)
    df["game_length_second"] = df["game_length_second"].astype(int)
    df["vision_wards_bought_in_game"] = df["vision_wards_bought_in_game"].astype(int)
    df["vision_score"] = df["vision_score"].astype(int)
    df["wards_placed"] = df["wards_placed"].astype(int)
    df["wards_killed"] = df["wards_killed"].astype(int)
    df["heal"] = df["heal"].astype(int)
    df["time_ccing_others"] = df["time_ccing_others"].astype(int)
    df["damage_dealt_to_champions"] = df["damage_dealt_to_champions"].astype(int)
    
    return df

def process_match_data(df):
    df = "test"
    return df