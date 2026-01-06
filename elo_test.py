import pandas as pd
import numpy as np

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("data/nba_games_last_5_years.csv")

# chronological index (same as earlier phases)
df = df.reset_index(drop=True)
df["game_id"] = df.index

# -------------------------------------------------
# ELO PARAMETERS
# -------------------------------------------------

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 0  # keep 0 for now; we’ll test later

# -------------------------------------------------
# INITIALIZE TEAM ELOs
# -------------------------------------------------

teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
elo = {team: INITIAL_ELO for team in teams}

# -------------------------------------------------
# ELO FUNCTIONS
# -------------------------------------------------

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


# -------------------------------------------------
# COMPUTE ELO FEATURES
# -------------------------------------------------

home_elos = []
away_elos = []

for _, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    home_elo = elo[home] + HOME_ADVANTAGE
    away_elo = elo[away]

    home_elos.append(home_elo)
    away_elos.append(away_elo)

    # expected result
    exp_home = expected_score(home_elo, away_elo)

    # actual result
    actual_home = row["home_win"]

    # update ELOs
    elo[home] += K_FACTOR * (actual_home - exp_home)
    elo[away] += K_FACTOR * ((1 - actual_home) - (1 - exp_home))


df["home_elo"] = home_elos
df["away_elo"] = away_elos
df["elo_diff"] = df["home_elo"] - df["away_elo"]

print(df[["home_team", "away_team", "home_elo", "away_elo", "elo_diff"]].head())
print("ELO computation complete")
print(df.tail()[["home_team", "away_team", "home_elo", "away_elo", "elo_diff"]])
print(df["elo_diff"].describe())
