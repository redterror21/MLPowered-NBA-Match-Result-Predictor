import pandas as pd

# -------------------------------------------------
# PHASE 1: LOAD DATA AND FIX TIME ORDER
# -------------------------------------------------

# Load dataset
df = pd.read_csv("data/nba_games_last_5_years.csv")

# Create a chronological index (safe replacement for date)
df = df.reset_index(drop=True)
df["game_id"] = df.index

print("Total games:", len(df))
print(df.head())

# Train / validation split
train_df = df[df["season"] <= 2024].copy()
val_df   = df[df["season"] == 2025].copy()

print("Train games:", len(train_df))
print("Validation games:", len(val_df))


# -------------------------------------------------
# PHASE 2: FEATURE ENGINEERING (ROLLING WIN %)
# -------------------------------------------------

def add_team_win_pct(df):
    """
    Creates rolling win percentage (last 10 games) for each team.
    Uses game_id to preserve chronological order.
    """
    df = df.sort_values("game_id")

    records = []

    for _, row in df.iterrows():
        # Home team result
        records.append({
            "game_id": row["game_id"],
            "team": row["home_team"],
            "win": row["home_win"]
        })

        # Away team result
        records.append({
            "game_id": row["game_id"],
            "team": row["away_team"],
            "win": 1 - row["home_win"]
        })

    team_df = pd.DataFrame(records)

    team_df["rolling_win_pct"] = (
        team_df
        .groupby("team")["win"]
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

print("\nTrain feature sample:")
print(train_df[["home_win_pct", "away_win_pct"]].head())


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

print("\nValidation feature sample:")
print(val_df[["home_win_pct", "away_win_pct"]].head())


# -------------------------------------------------
# FINAL SANITY CHECKS
# -------------------------------------------------

print("\nTrain feature stats:")
print(train_df[["home_win_pct", "away_win_pct"]].describe())

print("\nValidation feature stats:")
print(val_df[["home_win_pct", "away_win_pct"]].describe())

print("\nPhase 1 and 2 completed successfully.")
