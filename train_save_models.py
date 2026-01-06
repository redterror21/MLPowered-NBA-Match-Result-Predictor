import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

# =================================================
# LOAD DATA
# =================================================

df = pd.read_csv("data/nba_games_last_5_years.csv")
df = df.reset_index(drop=True)
df["game_id"] = df.index

train_df = df[df["season"] <= 2024].copy()
val_df   = df[df["season"] == 2025].copy()

# =================================================
# ROLLING WIN %
# =================================================

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


# ---- build rolling features ----

team_stats_train = add_team_win_pct(train_df)
team_stats_all   = add_team_win_pct(df[df["season"] <= 2025])

for side in ["home", "away"]:
    train_df = train_df.merge(
        team_stats_train[["game_id", "team", "rolling_win_pct"]],
        left_on=["game_id", f"{side}_team"],
        right_on=["game_id", "team"],
        how="left"
    ).rename(columns={"rolling_win_pct": f"{side}_win_pct"}).drop(columns=["team"])

    val_df = val_df.merge(
        team_stats_all[["game_id", "team", "rolling_win_pct"]],
        left_on=["game_id", f"{side}_team"],
        right_on=["game_id", "team"],
        how="left"
    ).rename(columns={"rolling_win_pct": f"{side}_win_pct"}).drop(columns=["team"])

train_df[["home_win_pct", "away_win_pct"]] = train_df[["home_win_pct", "away_win_pct"]].fillna(0.5)
val_df[["home_win_pct", "away_win_pct"]]   = val_df[["home_win_pct", "away_win_pct"]].fillna(0.5)

# =================================================
# ELO (HOME ADVANTAGE + SEASON RESET)
# =================================================

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

    h_pre = elo[h] + HOME_ADVANTAGE
    a_pre = elo[a]

    home_elos.append(h_pre)
    away_elos.append(a_pre)

    exp_h = expected(h_pre, a_pre)
    res = row["home_win"]

    elo[h] += K * (res - exp_h)
    elo[a] += K * ((1 - res) - (1 - exp_h))

df["elo_diff"] = np.array(home_elos) - np.array(away_elos)

train_df = train_df.merge(df[["game_id", "elo_diff"]], on="game_id", how="left")
val_df   = val_df.merge(df[["game_id", "elo_diff"]], on="game_id", how="left")

# =================================================
# FEATURE MATRIX
# =================================================

FEATURES = ["elo_diff", "home_win_pct", "away_win_pct"]

X_train = train_df[FEATURES].values
y_train = train_df["home_win"].values

# =================================================
# MODEL 1 — BATCH LOGISTIC REGRESSION
# =================================================

scaler_batch = StandardScaler()
X_train_batch = scaler_batch.fit_transform(X_train)

batch_model = LogisticRegression(max_iter=1000)
batch_model.fit(X_train_batch, y_train)

joblib.dump(batch_model, "model_batch.pkl")
joblib.dump(scaler_batch, "scaler_batch.pkl")

print("Saved batch model")

# =================================================
# MODEL 2 — INCREMENTAL SGD
# =================================================

scaler_inc = StandardScaler()
X_train_inc = scaler_inc.fit_transform(X_train)

incremental_model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.001,
    alpha=0.001,
    max_iter=1,
    tol=None,
    random_state=42
)

incremental_model.partial_fit(X_train_inc, y_train, classes=[0, 1])

joblib.dump(incremental_model, "model_incremental.pkl")
joblib.dump(scaler_inc, "scaler_incremental.pkl")

print("Saved incremental model")

# =================================================
# DONE
# =================================================

print("\nALL MODELS TRAINED AND SAVED")
print("Files created:")
print(" - model_batch.pkl")
print(" - scaler_batch.pkl")
print(" - model_incremental.pkl")
print(" - scaler_incremental.pkl")
