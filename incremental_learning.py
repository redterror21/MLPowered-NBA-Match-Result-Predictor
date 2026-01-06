import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv("data/nba_games_last_5_years.csv")
df = df.reset_index(drop=True)
df["game_id"] = df.index

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

# -------------------------------------------------
# ELO (HOME ADV + SEASON RESET)
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

# -------------------------------------------------
# INCREMENTAL MODEL (STABLE)
# -------------------------------------------------

FEATURES = ["elo_diff", "home_win_pct", "away_win_pct"]

X_train = train_df[FEATURES].values
y_train = train_df["home_win"].values

X_val = val_df[FEATURES].values
y_val = val_df["home_win"].values

# scale features (CRITICAL)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

model = SGDClassifier(
    loss="log_loss",
    learning_rate="constant",
    eta0=0.001,
    alpha=0.001,
    max_iter=1,
    tol=None,
    random_state=42
)

# initial training
model.partial_fit(X_train, y_train, classes=[0, 1])

# -------------------------------------------------
# ONLINE UPDATES
# -------------------------------------------------

probs = []

for i in range(len(X_val)):
    x = X_val[i].reshape(1, -1)
    y = y_val[i]

    prob = model.predict_proba(x)[0, 1]
    probs.append(prob)

    model.partial_fit(x, [y])

# -------------------------------------------------
# EVALUATION
# -------------------------------------------------

print("\nPHASE 6D — INCREMENTAL LEARNING (FIXED)")
print("Log Loss:", log_loss(y_val, probs))
print("Brier Score:", brier_score_loss(y_val, probs))
print("Avg predicted prob:", np.mean(probs))
print("Actual home win rate:", y_val.mean())
