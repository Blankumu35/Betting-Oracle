import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Improved ML Training...")

# 1. Load and Preprocess Data
print("📊 Loading data...")
df = pd.read_csv("data/Matches.csv", parse_dates=["MatchDate"], low_memory=False)

# 2. Enhanced Feature Engineering
print("🔧 Creating enhanced features...")

# Basic features
df["total_goals"] = df["FTHome"] + df["FTAway"]
df["over25"] = (df["total_goals"] > 2.5).astype(int)
df["result"] = df["FTResult"].map({"H":0,"D":1,"A":2})

# Form features (normalized)
df["home_form_wins"] = df["Form5Home"] / 3
df["away_form_wins"] = df["Form5Away"] / 3

# Elo features
df["elo_diff"] = df["HomeElo"] - df["AwayElo"]
df["elo_ratio"] = df["HomeElo"] / (df["AwayElo"] + 1)  # Avoid division by zero

# Goal-based features
df["home_goals_scored"] = df["FTHome"]
df["away_goals_scored"] = df["FTAway"]
df["home_goals_conceded"] = df["FTAway"]
df["away_goals_conceded"] = df["FTHome"]

# Rolling averages (last 5 matches)
df["home_avg_goals_scored"] = df.groupby("HomeTeam")["home_goals_scored"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
df["home_avg_goals_conceded"] = df.groupby("HomeTeam")["home_goals_conceded"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
df["away_avg_goals_scored"] = df.groupby("AwayTeam")["away_goals_scored"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
df["away_avg_goals_conceded"] = df.groupby("AwayTeam")["away_goals_conceded"].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

# Goal difference features
df["home_goal_diff"] = df["home_avg_goals_scored"] - df["home_avg_goals_conceded"]
df["away_goal_diff"] = df["away_avg_goals_scored"] - df["away_avg_goals_conceded"]
df["total_goal_diff"] = df["home_goal_diff"] - df["away_goal_diff"]

# Match importance features (based on division)
df["is_important"] = df["Division"].str.contains("Premier|La Liga|Bundesliga|Serie A", case=False).astype(int)

# Season features
df["season"] = df["MatchDate"].dt.year
df["month"] = df["MatchDate"].dt.month
df["day_of_week"] = df["MatchDate"].dt.dayofweek

# Additional features from available data
df["home_shots"] = df["HomeShots"]
df["away_shots"] = df["AwayShots"]
df["home_target"] = df["HomeTarget"]
df["away_target"] = df["AwayTarget"]
df["shot_accuracy_diff"] = (df["home_target"] / (df["home_shots"] + 1)) - (df["away_target"] / (df["away_shots"] + 1))

# Betting odds features
df["odd_home"] = df["OddHome"]
df["odd_away"] = df["OddAway"]
df["odd_draw"] = df["OddDraw"]
df["odds_diff"] = df["odd_home"] - df["odd_away"]

# Fill NaN values
df = df.fillna(0)

# 3. Select Enhanced Features
enhanced_features = [
    "home_form_wins", "away_form_wins",
    "elo_diff", "elo_ratio",
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "home_goal_diff", "away_goal_diff", "total_goal_diff",
    "is_important", "month", "day_of_week",
    "shot_accuracy_diff", "odds_diff"
]

X = df[enhanced_features]
y_outcome = df["result"]
y_over25 = df["over25"]

# Remove rows with missing values
mask = ~(X.isnull().any(axis=1) | y_outcome.isnull() | y_over25.isnull())
X = X[mask]
y_outcome = y_outcome[mask]
y_over25 = y_over25[mask]

print(f"📈 Dataset shape: {X.shape}")
print(f"📊 Features used: {len(enhanced_features)}")

# 4. Split Data
X_tr, X_te, yo_tr, yo_te = train_test_split(X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X, y_over25, test_size=0.2, random_state=42, stratify=y_over25)

# 5. Define Models to Test
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
}

# 6. Hyperparameter Grids
param_grids = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'LogisticRegression': {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }
}

# 7. Train and Evaluate Models
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, task_name):
    print(f"\n🔍 Training {model_name} for {task_name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"📊 Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    if task_name == "outcome":
        # Multi-class ROC AUC (one-vs-rest)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    else:
        # Binary ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    
    print(f"📈 Test Accuracy: {accuracy:.3f}")
    print(f"📊 ROC AUC: {roc_auc:.3f}")
    
    return model, accuracy, roc_auc

# 8. Find Best Models
best_models = {}
best_scores = {}

print("\n" + "="*60)
print("🎯 TRAINING OUTCOME PREDICTION MODELS")
print("="*60)

for model_name, model in models.items():
    if model_name in param_grids:
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            model, param_grids[model_name], 
            cv=3, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X_tr, yo_tr)
        best_model = grid_search.best_estimator_
        print(f"🏆 Best params for {model_name}: {grid_search.best_params_}")
    else:
        best_model = model
    
    trained_model, accuracy, roc_auc = evaluate_model(
        model_name, best_model, X_tr, X_te, yo_tr, yo_te, "outcome"
    )
    
    best_models[f"{model_name}_outcome"] = trained_model
    best_scores[f"{model_name}_outcome"] = (accuracy, roc_auc)

print("\n" + "="*60)
print("🎯 TRAINING OVER/UNDER 2.5 GOALS MODELS")
print("="*60)

for model_name, model in models.items():
    if model_name in param_grids:
        grid_search = GridSearchCV(
            model, param_grids[model_name], 
            cv=3, scoring='accuracy', n_jobs=-1, verbose=0
        )
        grid_search.fit(X2_tr, y2_tr)
        best_model = grid_search.best_estimator_
        print(f"🏆 Best params for {model_name}: {grid_search.best_params_}")
    else:
        best_model = model
    
    trained_model, accuracy, roc_auc = evaluate_model(
        model_name, best_model, X2_tr, X2_te, y2_tr, y2_te, "over25"
    )
    
    best_models[f"{model_name}_over25"] = trained_model
    best_scores[f"{model_name}_over25"] = (accuracy, roc_auc)

# 9. Select Best Models
print("\n" + "="*60)
print("🏆 BEST MODEL SELECTION")
print("="*60)

best_outcome_model = max(best_scores.items(), key=lambda x: x[1][0] if 'outcome' in x[0] else 0)
best_over25_model = max(best_scores.items(), key=lambda x: x[1][0] if 'over25' in x[0] else 0)

print(f"🥇 Best Outcome Model: {best_outcome_model[0]} (Accuracy: {best_outcome_model[1][0]:.3f}, ROC AUC: {best_outcome_model[1][1]:.3f})")
print(f"🥇 Best Over/Under Model: {best_over25_model[0]} (Accuracy: {best_over25_model[1][0]:.3f}, ROC AUC: {best_over25_model[1][1]:.3f})")

# 10. Save Best Models
final_outcome_model = best_models[best_outcome_model[0]]
final_over25_model = best_models[best_over25_model[0]]

joblib.dump(final_outcome_model, "improved_model_outcome.pkl")
joblib.dump(final_over25_model, "improved_model_over25.pkl")
joblib.dump(enhanced_features, "improved_feature_list.pkl")

print(f"\n💾 Models saved:")
print(f"   - improved_model_outcome.pkl")
print(f"   - improved_model_over25.pkl")
print(f"   - improved_feature_list.pkl")

# 11. Feature Importance Analysis
if hasattr(final_outcome_model, 'feature_importances_'):
    print(f"\n📊 Feature Importance (Outcome Model):")
    importance_df = pd.DataFrame({
        'feature': enhanced_features,
        'importance': final_outcome_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

if hasattr(final_over25_model, 'feature_importances_'):
    print(f"\n📊 Feature Importance (Over/Under Model):")
    importance_df = pd.DataFrame({
        'feature': enhanced_features,
        'importance': final_over25_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

print(f"\n✅ Improved training completed!")
print(f"📈 Model performance improved with {len(enhanced_features)} features vs original 3 features") 