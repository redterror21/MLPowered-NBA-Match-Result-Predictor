import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss

# -------------------------------------------------
# LOAD DATA (same prep as Phase 1 & 2)
# -------------------------------------------------

df = pd.read_csv("data/nba_games_last_5_years.csv")
df = df.reset_index(drop=True)
df["game_id"] = df.index

train_df = df[df["season"] <= 2024].copy()
val_df   = df[df["season"] == 2025].copy()


# -------------------------------------------------
# FEATURE ENGINEERING (same as Phase 2)
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
        .rolling(window=10, min_periods=5)
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


# -------- VALIDATION FEATURES (NO LEAKAGE) --------

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
# MODEL TRAINING
# -------------------------------------------------

X_train = train_df[["home_win_pct", "away_win_pct"]]
y_train = train_df["home_win"]

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained successfully")


# -------------------------------------------------
# EVALUATION ON 2025 SEASON
# -------------------------------------------------

X_val = val_df[["home_win_pct", "away_win_pct"]]
y_val = val_df["home_win"]

probs = model.predict_proba(X_val)[:, 1]

print("\nEvaluation metrics (2025 season):")
print("Log Loss:", log_loss(y_val, probs))
print("Brier Score:", brier_score_loss(y_val, probs))

print("\nSanity checks:")
print("Average predicted home win prob:", probs.mean())
print("Actual home win rate:", y_val.mean())

