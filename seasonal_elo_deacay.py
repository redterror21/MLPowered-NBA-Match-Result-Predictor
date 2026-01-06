import pandas as pd

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("data/nba_games_last_5_years.csv")

# chronological index
df = df.reset_index(drop=True)
df["game_id"] = df.index

# -------------------------------------------------
# ELO PARAMETERS
# -------------------------------------------------

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 75
RESET_WEIGHT = 0.25   # portion pulled back to 1500 each season

# -------------------------------------------------
# INITIALIZE TEAM ELOS
# -------------------------------------------------

teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
elo = {team: INITIAL_ELO for team in teams}

# -------------------------------------------------
# ELO FUNCTION
# -------------------------------------------------

def expected_score(elo_home, elo_away):
    return 1 / (1 + 10 ** ((elo_away - elo_home) / 400))

# -------------------------------------------------
# COMPUTE ELO WITH SEASON RESET
# -------------------------------------------------

home_elos = []
away_elos = []

current_season = None

for _, row in df.iterrows():

    # ----- SEASON RESET -----
    if current_season is None:
        current_season = row["season"]

    if row["season"] != current_season:
        for team in elo:
            elo[team] = (1 - RESET_WEIGHT) * elo[team] + RESET_WEIGHT * INITIAL_ELO
        current_season = row["season"]

    home = row["home_team"]
    away = row["away_team"]

    # pre-game ELOs (home advantage only for expectation)
    home_elo_pre = elo[home] + HOME_ADVANTAGE
    away_elo_pre = elo[away]

    home_elos.append(home_elo_pre)
    away_elos.append(away_elo_pre)

    exp_home = expected_score(home_elo_pre, away_elo_pre)
    actual_home = row["home_win"]

    # update BASE ELOs
    elo[home] += K_FACTOR * (actual_home - exp_home)
    elo[away] += K_FACTOR * ((1 - actual_home) - (1 - exp_home))

# -------------------------------------------------
# SAVE FEATURES
# -------------------------------------------------

df["home_elo"] = home_elos
df["away_elo"] = away_elos
df["elo_diff"] = df["home_elo"] - df["away_elo"]

print(df[["home_team", "away_team", "home_elo", "away_elo", "elo_diff"]].head())
print("\nELO with home advantage + seasonal reset computed")

print("\nELO diff stats:")
print(df["elo_diff"].describe())
