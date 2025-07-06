# 1. Load Data
import pandas as pd

# Load the data
df = pd.read_csv("data/Matches.csv", parse_dates=["MatchDate"])

# 2. Feature Engineering
df["total_goals"] = df["FTHome"] + df["FTAway"]
df["over25"] = (df["total_goals"] > 2.5).astype(int)
df["result"] = df["FTResult"].map({"H":0,"D":1,"A":2})

# Use provided form columns
df["home_form_wins"] = df["Form5Home"] / 3  # normalize to W count
df["away_form_wins"] = df["Form5Away"] / 3

# Elo features
df["elo_diff"] = df["HomeElo"] - df["AwayElo"]

# Optional: head-to-head computed separately
# df["h2h_avg_goals"] = ... 

# 3. Select Features & Split
features = ["home_form_wins", "away_form_wins","elo_diff"]
# Drop rows with missing values in features or targets
df = df.dropna(subset=features + ["result", "over25"])
X = df[features]
y_outcome = df["result"]
y_over25 = df["over25"]

from sklearn.model_selection import train_test_split
X_tr, X_te, yo_tr, yo_te = train_test_split(X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X, y_over25, test_size=0.2, random_state=42, stratify=y_over25)

# 4. Train Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

model_outcome = RandomForestClassifier(n_estimators=200, random_state=42)
model_outcome.fit(X_tr, yo_tr)
print(classification_report(yo_te, model_outcome.predict(X_te)))

model_over = RandomForestClassifier(n_estimators=200, random_state=42)
model_over.fit(X2_tr, y2_tr)
print(classification_report(y2_te, model_over.predict(X2_te)))

# 5. Save Models and Feature List
joblib.dump(model_outcome, "model_outcome.pkl")
joblib.dump(model_over, "model_over25.pkl")
joblib.dump(features, "feature_list.pkl") 