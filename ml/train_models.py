import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. Load Data
df = pd.read_csv("data/Matches.csv", parse_dates=["MatchDate"])

# 2. Define ONLY the H2H feature list
h2h_feature_list = [
    "Home_Form", "Home_Last6_Matches", "Home_Last6_Goals", "Home_Last6_per game", "Home_Last6_Wins",
    "Home_Last6_Draws", "Home_Last6_Losses", "Home_Last6_Over 2.5", "Home_Last6_Over 1.5", "Home_Last6_CS", "Home_Last6_BTTS",
    "Home_Overall_Matches", "Home_Overall_Goals", "Home_Overall_per game", "Home_Overall_Wins", "Home_Overall_Draws",
    "Home_Overall_Losses", "Home_Overall_Over 2.5", "Home_Overall_Over 1.5", "Home_Overall_CS", "Home_Overall_BTTS", "Away_Last6_Matches", "Away_Last6_Goals", "Away_Last6_per game", "Away_Last6_Wins",
    "Away_Last6_Draws", "Away_Last6_Losses", "Away_Last6_Over 2.5", "Away_Last6_Over 1.5", "Away_Last6_CS", "Away_Last6_BTTS",
    "Away_Overall_Matches", "Away_Overall_Goals", "Away_Overall_per game", "Away_Overall_Wins", "Away_Overall_Draws",
    "Away_Overall_Losses", "Away_Overall_Over 2.5", "Away_Overall_Over 1.5", "Away_Overall_CS", "Away_Overall_BTTS"
]

# 3. Ensure all H2H columns exist in the DataFrame
for col in h2h_feature_list:
    if col not in df.columns:
        df[col] = 0

# Convert form strings to numeric counts (always create columns)
for prefix in ["Home", "Away"]:
    form_col = f"{prefix}_Form"
    df[f"{prefix}_Form_W"] = df[form_col].apply(lambda x: str(x).count("W") if pd.notnull(x) else 0) if form_col in df.columns else 0
    df[f"{prefix}_Form_D"] = df[form_col].apply(lambda x: str(x).count("D") if pd.notnull(x) else 0) if form_col in df.columns else 0
    df[f"{prefix}_Form_L"] = df[form_col].apply(lambda x: str(x).count("L") if pd.notnull(x) else 0) if form_col in df.columns else 0

# Remove original string form columns from feature list
h2h_numeric_features = [
    "Home_Last6_Matches", "Home_Last6_Goals", "Home_Last6_per game", "Home_Last6_Wins",
    "Home_Last6_Draws", "Home_Last6_Losses", "Home_Last6_Over 2.5", "Home_Last6_Over 1.5", "Home_Last6_CS", "Home_Last6_BTTS",
    "Home_Overall_Matches", "Home_Overall_Goals", "Home_Overall_per game", "Home_Overall_Wins", "Home_Overall_Draws",
    "Home_Overall_Losses", "Home_Overall_Over 2.5", "Home_Overall_Over 1.5", "Home_Overall_CS", "Home_Overall_BTTS",
    "Away_Last6_Matches", "Away_Last6_Goals", "Away_Last6_per game", "Away_Last6_Wins",
    "Away_Last6_Draws", "Away_Last6_Losses", "Away_Last6_Over 2.5", "Away_Last6_Over 1.5", "Away_Last6_CS", "Away_Last6_BTTS",
    "Away_Overall_Matches", "Away_Overall_Goals", "Away_Overall_per game", "Away_Overall_Wins", "Away_Overall_Draws",
    "Away_Overall_Losses", "Away_Overall_Over 2.5", "Away_Overall_Over 1.5", "Away_Overall_CS", "Away_Overall_BTTS",
    "Home_Form_W", "Home_Form_D", "Home_Form_L", "Away_Form_W", "Away_Form_D", "Away_Form_L"
]

# 4. Prepare target columns
df["total_goals"] = df["FTHome"] + df["FTAway"]
df["over25"] = (df["total_goals"] > 2.5).astype(int)
df["result"] = df["FTResult"].map({"H":0,"D":1,"A":2})

# 5. Drop rows with missing values in features or targets
df = df.dropna(subset=h2h_numeric_features + ["result", "over25"])

# Convert percentage strings to decimals for all relevant columns
for col in [
    "Home_Overall_Over 2.5", "Home_Overall_Over 1.5", "Home_Overall_CS", "Home_Overall_BTTS",
    "Away_Overall_Over 2.5", "Away_Overall_Over 1.5", "Away_Overall_CS", "Away_Overall_BTTS"
]:
    if col in df.columns:
        # Remove '%' and divide by 100 to get decimal
        df[col] = df[col].replace(r'%', '', regex=True).astype(float) / 100.0

# 6. Select features and targets
X = df[h2h_numeric_features]
y_outcome = df["result"]
y_over25 = df["over25"]

# 7. Train/Test Split
X_tr, X_te, yo_tr, yo_te = train_test_split(X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X, y_over25, test_size=0.2, random_state=42, stratify=y_over25)

# 8. Train Models
model_outcome = RandomForestClassifier(n_estimators=200, random_state=42)
model_outcome.fit(X_tr, yo_tr)
print("Outcome Model Report:")
print(classification_report(yo_te, model_outcome.predict(X_te)))

model_over = RandomForestClassifier(n_estimators=200, random_state=42)
model_over.fit(X2_tr, y2_tr)
print("Over/Under Model Report:")
print(classification_report(y2_te, model_over.predict(X2_te)))

# 9. Save Models and Feature List
joblib.dump(model_outcome, "model_outcome.pkl")
joblib.dump(model_over, "model_over25.pkl")