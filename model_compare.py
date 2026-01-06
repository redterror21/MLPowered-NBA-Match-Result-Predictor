import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("data/nba_games_last_5_years.csv")
df = df.reset_index(drop=True)
df["game_id"] = df.index

# -------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------------------------

train_df = df[df["season"] <= 2024].copy()
val_df   = df[df["season"] == 2025].copy()

# -------------------------------------------------
# ROLLING WIN %
# -------------------------------------------------

def add_team_win_pct(df):
    df = df.sort_values("game_id")
    records = []

    for _, row in df.iterrows():
        records.append({
            "game_id": row["game_id"],
            "team": row["home_team"],
            "win": row["home_win"]
        })
        records.append({
            "game_id": row["game_id"],
            "team": row["away_team"],
            "win": 1 - row["home_win"]
        })

    team_df = pd.DataFrame(records)

    team_df["rolling_win_pct"] = (
        team_df.groupby("team")["win"]
        .rolling(10, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return team_df


# -------- TRAIN FEATURES --------

team_stats_train = add_team_win_pct(train_df)

for side in ["home", "away"]:
    train_df = train_df.merge(
        team_stats_train[["game_id", "team", "rolling_win_pct"]],
        left_on=["game_id", f"{side}_team"],
        right_on=["game_id", "team"],
        how="left"
    ).rename(columns={"rolling_win_pct": f"{side}_win_pct"}).drop(columns=["team"])

train_df[["home_win_pct", "away_win_pct"]] = (
    train_df[["home_win_pct", "away_win_pct"]].fillna(0.5)
)

# -------- VALIDATION FEATURES --------

combined_df = df[df["season"] <= 2025].copy()
team_stats_all = add_team_win_pct(combined_df)

for side in ["home", "away"]:
    val_df = val_df.merge(
        team_stats_all[["game_id", "team", "rolling_win_pct"]],
        left_on=["game_id", f"{side}_team"],
        right_on=["game_id", "team"],
        how="left"
    ).rename(columns={"rolling_win_pct": f"{side}_win_pct"}).drop(columns=["team"])

val_df[["home_win_pct", "away_win_pct"]] = (
    val_df[["home_win_pct", "away_win_pct"]].fillna(0.5)
)

# -------------------------------------------------
# ELO WITH HOME ADVANTAGE + SEASON RESET (6B)
# -------------------------------------------------

INITIAL_ELO = 1500
K = 20
HOME_ADVANTAGE = 75
RESET_WEIGHT = 0.25

teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
elo = {team: INITIAL_ELO for team in teams}

home_elos, away_elos = [], []
current_season = None

def expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

for _, row in df.iterrows():

    if current_season is None:
        current_season = row["season"]

    if row["season"] != current_season:
        for t in elo:
            elo[t] = (1 - RESET_WEIGHT) * elo[t] + RESET_WEIGHT * INITIAL_ELO
        current_season = row["season"]

    h, a = row["home_team"], row["away_team"]

    h_elo_pre = elo[h] + HOME_ADVANTAGE
    a_elo_pre = elo[a]

    home_elos.append(h_elo_pre)
    away_elos.append(a_elo_pre)

    exp_h = expected(h_elo_pre, a_elo_pre)
    res = row["home_win"]

    elo[h] += K * (res - exp_h)
    elo[a] += K * ((1 - res) - (1 - exp_h))

df["home_elo"] = home_elos
df["away_elo"] = away_elos
df["elo_diff"] = df["home_elo"] - df["away_elo"]

# Merge ELO into train/val
for d in [train_df, val_df]:
    d.merge(
        df[["game_id", "home_elo", "away_elo", "elo_diff"]],
        on="game_id",
        how="left"
    )

train_df = train_df.merge(
    df[["game_id", "home_elo", "away_elo", "elo_diff"]],
    on="game_id",
    how="left"
)

val_df = val_df.merge(
    df[["game_id", "home_elo", "away_elo", "elo_diff"]],
    on="game_id",
    how="left"
)

# -------------------------------------------------
# MODEL COMPARISON
# -------------------------------------------------

models = {
    "ELO only": ["elo_diff"],
    "Rolling form only": ["home_win_pct", "away_win_pct"],
    "ELO + form": ["elo_diff", "home_win_pct", "away_win_pct"]
}

print("\nPHASE 6C — MODEL COMPARISON\n")

for name, features in models.items():
    model = LogisticRegression(max_iter=1000)
    model.fit(train_df[features], train_df["home_win"])

    probs = model.predict_proba(val_df[features])[:, 1]

    print(f"{name}")
    print("  Log Loss:", log_loss(val_df["home_win"], probs))
    print("  Brier:", brier_score_loss(val_df["home_win"], probs))
    print("  Avg prob:", probs.mean())
    print()
