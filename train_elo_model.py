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
# SPLIT
# -------------------------------------------------

train_df = df[df["season"] <= 2024].copy()
val_df   = df[df["season"] == 2025].copy()

# -------------------------------------------------
# ROLLING WIN % (same as earlier)
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

train_df = train_df.merge(
    team_stats_train[["game_id", "team", "rolling_win_pct"]],
    left_on=["game_id", "home_team"],
    right_on=["game_id", "team"],
    how="left"
).rename(columns={"rolling_win_pct": "home_win_pct"}).drop(columns=["team"])

train_df = train_df.merge(
    team_stats_train[["game_id", "team", "rolling_win_pct"]],
    left_on=["game_id", "away_team"],
    right_on=["game_id", "team"],
    how="left"
).rename(columns={"rolling_win_pct": "away_win_pct"}).drop(columns=["team"])

train_df[["home_win_pct", "away_win_pct"]] = (
    train_df[["home_win_pct", "away_win_pct"]].fillna(0.5)
)

# -------- VALIDATION FEATURES --------

combined_df = df[df["season"] <= 2025].copy()
team_stats_all = add_team_win_pct(combined_df)

val_df = val_df.merge(
    team_stats_all[["game_id", "team", "rolling_win_pct"]],
    left_on=["game_id", "home_team"],
    right_on=["game_id", "team"],
    how="left"
).rename(columns={"rolling_win_pct": "home_win_pct"}).drop(columns=["team"])

val_df = val_df.merge(
    team_stats_all[["game_id", "team", "rolling_win_pct"]],
    left_on=["game_id", "away_team"],
    right_on=["game_id", "team"],
    how="left"
).rename(columns={"rolling_win_pct": "away_win_pct"}).drop(columns=["team"])

val_df[["home_win_pct", "away_win_pct"]] = (
    val_df[["home_win_pct", "away_win_pct"]].fillna(0.5)
)

# -------------------------------------------------
# ADD ELO FEATURES (from Phase 4)
# -------------------------------------------------

# Recompute ELO inline to ensure no leakage
INITIAL_ELO = 1500
K = 20

teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
elo = {team: INITIAL_ELO for team in teams}

home_elos, away_elos = [], []

def expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

for _, row in df.iterrows():
    h, a = row["home_team"], row["away_team"]
    h_elo, a_elo = elo[h], elo[a]

    home_elos.append(h_elo)
    away_elos.append(a_elo)

    exp_h = expected(h_elo, a_elo)
    res = row["home_win"]

    elo[h] += K * (res - exp_h)
    elo[a] += K * ((1 - res) - (1 - exp_h))

df["home_elo"] = home_elos
df["away_elo"] = away_elos
df["elo_diff"] = df["home_elo"] - df["away_elo"]

# Merge ELO into train/val
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
# MODEL TRAINING
# -------------------------------------------------

FEATURES = [
    "home_elo", "away_elo", "elo_diff",
    "home_win_pct", "away_win_pct"
]

X_train = train_df[FEATURES]
y_train = train_df["home_win"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------

X_val = val_df[FEATURES]
y_val = val_df["home_win"]

probs = model.predict_proba(X_val)[:, 1]

print("\nPHASE 5 — ELO + FORM MODEL")
print("Log Loss:", log_loss(y_val, probs))
print("Brier Score:", brier_score_loss(y_val, probs))
print("Avg predicted home win prob:", probs.mean())
print("Actual home win rate:", y_val.mean())
