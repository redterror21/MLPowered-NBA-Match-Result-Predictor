import pandas as pd
import numpy as np
import joblib
from collections import defaultdict, deque
from sklearn.metrics import log_loss, brier_score_loss

# =================================================
# LOAD DATA
# =================================================

df = pd.read_csv("data/nba_games_last_5_years.csv")
df = df.reset_index(drop=True)

train_df = df[df["season"] <= 2024].copy()
test_df  = df[df["season"] == 2025].copy()

# =================================================
# LOAD MODEL
# =================================================

model = joblib.load("model_incremental.pkl")
scaler = joblib.load("scaler_incremental.pkl")

# =================================================
# PARAMETERS
# =================================================

INITIAL_ELO = 1500
HOME_ADVANTAGE = 75
K = 20
ROLLING_WINDOW = 10

# =================================================
# INITIAL STATE (END OF 2024)
# =================================================

elo = defaultdict(lambda: INITIAL_ELO)
recent_results = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))

# --- rebuild ELO + rolling form using train data ---

def expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

for _, row in train_df.iterrows():
    h, a = row["home_team"], row["away_team"]
    res = row["home_win"]

    h_pre = elo[h] + HOME_ADVANTAGE
    a_pre = elo[a]

    exp_h = expected(h_pre, a_pre)

    elo[h] += K * (res - exp_h)
    elo[a] += K * ((1 - res) - (1 - exp_h))

    recent_results[h].append(res)
    recent_results[a].append(1 - res)

# =================================================
# WALK-FORWARD THROUGH 2025
# =================================================

preds = []
probs = []
actuals = []

correct = 0

for i, row in test_df.iterrows():
    h, a = row["home_team"], row["away_team"]
    actual = row["home_win"]

    # --- features BEFORE game ---
    home_elo = elo[h] + HOME_ADVANTAGE
    away_elo = elo[a]

    home_form = np.mean(recent_results[h]) if recent_results[h] else 0.5
    away_form = np.mean(recent_results[a]) if recent_results[a] else 0.5

    X = np.array([[home_elo - away_elo, home_form, away_form]])
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0, 1]
    pred = 1 if prob >= 0.5 else 0

    # --- record ---
    probs.append(prob)
    preds.append(pred)
    actuals.append(actual)

    if pred == actual:
        correct += 1

    # --- UPDATE SYSTEM AFTER RESULT ---
    exp_h = expected(home_elo, away_elo)

    elo[h] += K * (actual - exp_h)
    elo[a] += K * ((1 - actual) - (1 - exp_h))

    recent_results[h].append(actual)
    recent_results[a].append(1 - actual)

    model.partial_fit(X_scaled, [actual])

# =================================================
# RESULTS
# =================================================

accuracy = correct / len(test_df)

print("\nWALK-FORWARD RESULTS — 2025 SEASON")
print(f"Games evaluated: {len(test_df)}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Log Loss: {log_loss(actuals, probs):.3f}")
print(f"Brier Score: {brier_score_loss(actuals, probs):.3f}")

# baseline: always pick home team
baseline_acc = np.mean(test_df["home_win"])
print(f"Home-team baseline accuracy: {baseline_acc:.3f}")
print(f"Improvement over baseline: {accuracy - baseline_acc:.3f}")
