import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Enhanced ML Training with Advanced Features...")

# 1. Load and Preprocess Data
print("📊 Loading data...")
df = pd.read_csv("data/Matches.csv", parse_dates=["MatchDate"], low_memory=False)

# Use more data for better training
df = df.sample(frac=0.4, random_state=42)
print(f"📈 Using {len(df)} samples for training")

# Add new features for real H2H and Elo (assuming you have these columns in your data)
if 'h2h_wins' not in df.columns:
    df['h2h_wins'] = 0
if 'h2h_draws' not in df.columns:
    df['h2h_draws'] = 0
if 'h2h_losses' not in df.columns:
    df['h2h_losses'] = 0
if 'h2h_avg_goals_3' not in df.columns:
    df['h2h_avg_goals_3'] = 0
if 'h2h_avg_goals_4' not in df.columns:
    df['h2h_avg_goals_4'] = 0
if 'Home Elo' not in df.columns:
    df['Home Elo'] = 1500
if 'Away Elo' not in df.columns:
    df['Away Elo'] = 1500

df['elo_diff'] = df['Home Elo'] - df['Away Elo']
df['elo_ratio'] = df['Home Elo'] / (df['Away Elo'] + 1)

# 2. Advanced Feature Engineering
print("🔧 Creating advanced features...")

# Basic features
df["total_goals"] = df["FTHome"] + df["FTAway"]
df["over25"] = (df["total_goals"] > 2.5).astype(int)
df["result"] = df["FTResult"].map({"H":0,"D":1,"A":2})

# Enhanced form features
df["home_form_wins"] = df["Form5Home"] / 3
df["away_form_wins"] = df["Form5Away"] / 3
df["form_difference"] = df["home_form_wins"] - df["away_form_wins"]

# Advanced Elo features
df["elo_diff"] = df["HomeElo"] - df["AwayElo"]
df["elo_ratio"] = df["HomeElo"] / (df["AwayElo"] + 1)
df["elo_interaction"] = df["elo_diff"] * df["form_difference"]

# Goal-based features with rolling statistics
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

# Match context features
df["is_important"] = df["Division"].str.contains("Premier|La Liga|Bundesliga|Serie A|Champions", case=False).astype(int)
df["month"] = df["MatchDate"].dt.month
df["day_of_week"] = df["MatchDate"].dt.dayofweek
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Betting odds features
df["odds_diff"] = df["OddHome"] - df["OddAway"]
df["odds_ratio"] = df["OddHome"] / (df["OddAway"] + 1)
df["total_odds"] = df["OddHome"] + df["OddAway"]

# Advanced statistical features
df["home_win_rate"] = df.groupby("HomeTeam")["result"].rolling(10, min_periods=1).apply(lambda x: (x == 0).mean()).reset_index(0, drop=True)
df["away_win_rate"] = df.groupby("AwayTeam")["result"].rolling(10, min_periods=1).apply(lambda x: (x == 2).mean()).reset_index(0, drop=True)

# Goals per game features
df["home_goals_per_game"] = df.groupby("HomeTeam")["home_goals_scored"].rolling(10, min_periods=1).mean().reset_index(0, drop=True)
df["away_goals_per_game"] = df.groupby("AwayTeam")["away_goals_scored"].rolling(10, min_periods=1).mean().reset_index(0, drop=True)

# Volatility features
df["home_goals_std"] = df.groupby("HomeTeam")["home_goals_scored"].rolling(10, min_periods=1).std().reset_index(0, drop=True)
df["away_goals_std"] = df.groupby("AwayTeam")["away_goals_scored"].rolling(10, min_periods=1).std().reset_index(0, drop=True)

# Interaction features
df["form_elo_interaction"] = df["form_difference"] * df["elo_diff"]
df["goals_form_interaction"] = df["total_goal_diff"] * df["form_difference"]

# Fill NaN values
df = df.fillna(0)

# 3. Select Advanced Features
advanced_features = [
    "home_form_wins", "away_form_wins", "form_difference",
    "elo_diff", "elo_ratio", "elo_interaction",
    "home_avg_goals_scored", "home_avg_goals_conceded",
    "away_avg_goals_scored", "away_avg_goals_conceded",
    "home_goal_diff", "away_goal_diff", "total_goal_diff",
    "is_important", "month", "is_weekend",
    "odds_diff", "odds_ratio", "total_odds",
    "home_win_rate", "away_win_rate",
    "home_goals_per_game", "away_goals_per_game",
    "home_goals_std", "away_goals_std",
    "form_elo_interaction", "goals_form_interaction"
]

# Define the feature list for training
feature_list = [
    'home_form_wins', 'away_form_wins', 'form_difference',
    'home_avg_goals_scored', 'home_avg_goals_conceded',
    'away_avg_goals_scored', 'away_avg_goals_conceded',
    'home_goal_diff', 'away_goal_diff', 'total_goal_diff',
    'is_important', 'odds_diff', 'home_win_rate', 'away_win_rate',
    'xg_difference', 'total_expected_goals',
    # New H2H and Elo features
    'h2h_wins', 'h2h_draws', 'h2h_losses', 'h2h_avg_goals_3', 'h2h_avg_goals_4',
    'Home Elo', 'Away Elo', 'elo_diff', 'elo_ratio'
]

X = df[advanced_features]
y_outcome = df["result"]
y_over25 = df["over25"]

# Remove rows with missing values
mask = ~(X.isnull().any(axis=1) | y_outcome.isnull() | y_over25.isnull())
X = X[mask]
y_outcome = y_outcome[mask]
y_over25 = y_over25[mask]

print(f"📈 Final dataset shape: {X.shape}")
print(f"📊 Advanced features used: {len(advanced_features)}")

# 4. Split Data
X_tr, X_te, yo_tr, yo_te = train_test_split(X, y_outcome, test_size=0.2, random_state=42, stratify=y_outcome)
X2_tr, X2_te, y2_tr, y2_te = train_test_split(X, y_over25, test_size=0.2, random_state=42, stratify=y_over25)

# 5. Define Enhanced Models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=8, random_state=42),
    'LogisticRegression': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
    'SVM': SVC(probability=True, random_state=42, kernel='rbf')
}

# 6. Enhanced Training and Evaluation
def evaluate_enhanced_model(model_name, model, X_train, X_test, y_train, y_test, task_name):
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
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
    
    print(f"📈 Test Accuracy: {accuracy:.3f}")
    print(f"📊 ROC AUC: {roc_auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"📋 Confusion Matrix:\n{cm}")
    
    return model, accuracy, roc_auc

# 7. Train Individual Models
enhanced_models = {}
enhanced_scores = {}

print("\n" + "="*60)
print("🎯 TRAINING ENHANCED OUTCOME PREDICTION MODELS")
print("="*60)

for model_name, model in models.items():
    trained_model, accuracy, roc_auc = evaluate_enhanced_model(
        model_name, model, X_tr, X_te, yo_tr, yo_te, "outcome"
    )
    
    enhanced_models[f"{model_name}_outcome"] = trained_model
    enhanced_scores[f"{model_name}_outcome"] = (accuracy, roc_auc)

print("\n" + "="*60)
print("🎯 TRAINING ENHANCED OVER/UNDER 2.5 GOALS MODELS")
print("="*60)

for model_name, model in models.items():
    trained_model, accuracy, roc_auc = evaluate_enhanced_model(
        model_name, model, X2_tr, X2_te, y2_tr, y2_te, "over25"
    )
    
    enhanced_models[f"{model_name}_over25"] = trained_model
    enhanced_scores[f"{model_name}_over25"] = (accuracy, roc_auc)

# 8. Create Ensemble Models
print("\n" + "="*60)
print("🏆 CREATING ENSEMBLE MODELS")
print("="*60)

# Select top 3 models for each task
top_outcome_models = sorted([(k, v) for k, v in enhanced_scores.items() if 'outcome' in k], 
                           key=lambda x: x[1][0], reverse=True)[:3]
top_over25_models = sorted([(k, v) for k, v in enhanced_scores.items() if 'over25' in k], 
                          key=lambda x: x[1][0], reverse=True)[:3]

print(f"Top 3 Outcome Models: {[m[0] for m in top_outcome_models]}")
print(f"Top 3 Over/Under Models: {[m[0] for m in top_over25_models]}")

# Create ensemble models
outcome_ensemble = VotingClassifier(
    estimators=[(name, enhanced_models[name]) for name, _ in top_outcome_models],
    voting='soft'
)

over25_ensemble = VotingClassifier(
    estimators=[(name, enhanced_models[name]) for name, _ in top_over25_models],
    voting='soft'
)

# Train ensemble models
print("\n🔍 Training Outcome Ensemble...")
outcome_ensemble.fit(X_tr, yo_tr)
outcome_ensemble_pred = outcome_ensemble.predict(X_te)
outcome_ensemble_proba = outcome_ensemble.predict_proba(X_te)
outcome_ensemble_accuracy = accuracy_score(yo_te, outcome_ensemble_pred)
outcome_ensemble_roc = roc_auc_score(yo_te, outcome_ensemble_proba, multi_class='ovr')

print(f"📈 Ensemble Outcome Accuracy: {outcome_ensemble_accuracy:.3f}")
print(f"📊 Ensemble Outcome ROC AUC: {outcome_ensemble_roc:.3f}")

print("\n🔍 Training Over/Under Ensemble...")
over25_ensemble.fit(X2_tr, y2_tr)
over25_ensemble_pred = over25_ensemble.predict(X2_te)
over25_ensemble_proba = over25_ensemble.predict_proba(X2_te)
over25_ensemble_accuracy = accuracy_score(y2_te, over25_ensemble_pred)
over25_ensemble_roc = roc_auc_score(y2_te, over25_ensemble_proba[:, 1])

print(f"📈 Ensemble Over/Under Accuracy: {over25_ensemble_accuracy:.3f}")
print(f"📊 Ensemble Over/Under ROC AUC: {over25_ensemble_roc:.3f}")

# 9. Save Enhanced Models
print("\n" + "="*60)
print("💾 SAVING ENHANCED MODELS")
print("="*60)

# Save ensemble models
joblib.dump(outcome_ensemble, "enhanced_model_outcome.pkl")
joblib.dump(over25_ensemble, "enhanced_model_over25.pkl")
joblib.dump(advanced_features, "enhanced_feature_list.pkl")

# Save the feature list for inference
joblib.dump(feature_list, "fast_improved_feature_list.pkl")

# Save individual best models too
best_outcome_model = top_outcome_models[0][0]
best_over25_model = top_over25_models[0][0]

joblib.dump(enhanced_models[best_outcome_model], "enhanced_best_outcome.pkl")
joblib.dump(enhanced_models[best_over25_model], "enhanced_best_over25.pkl")

print(f"💾 Enhanced models saved:")
print(f"   - enhanced_model_outcome.pkl (Ensemble)")
print(f"   - enhanced_model_over25.pkl (Ensemble)")
print(f"   - enhanced_best_outcome.pkl (Best Individual)")
print(f"   - enhanced_best_over25.pkl (Best Individual)")
print(f"   - enhanced_feature_list.pkl")

# 10. Feature Importance Analysis
print("\n" + "="*60)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature importance from best individual models
best_outcome = enhanced_models[best_outcome_model]
best_over25 = enhanced_models[best_over25_model]

if hasattr(best_outcome, 'feature_importances_'):
    print(f"\n📊 Feature Importance (Best Outcome Model - {best_outcome_model}):")
    importance_df = pd.DataFrame({
        'feature': advanced_features,
        'importance': best_outcome.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

if hasattr(best_over25, 'feature_importances_'):
    print(f"\n📊 Feature Importance (Best Over/Under Model - {best_over25_model}):")
    importance_df = pd.DataFrame({
        'feature': advanced_features,
        'importance': best_over25.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

# 11. Performance Summary
print("\n" + "="*60)
print("🏆 FINAL PERFORMANCE SUMMARY")
print("="*60)

print(f"🥇 Best Individual Outcome Model: {best_outcome_model}")
print(f"   Accuracy: {enhanced_scores[best_outcome_model][0]:.3f}")
print(f"   ROC AUC: {enhanced_scores[best_outcome_model][1]:.3f}")

print(f"🥇 Best Individual Over/Under Model: {best_over25_model}")
print(f"   Accuracy: {enhanced_scores[best_over25_model][0]:.3f}")
print(f"   ROC AUC: {enhanced_scores[best_over25_model][1]:.3f}")

print(f"\n🏆 Ensemble Performance:")
print(f"   Outcome Accuracy: {outcome_ensemble_accuracy:.3f}")
print(f"   Outcome ROC AUC: {outcome_ensemble_roc:.3f}")
print(f"   Over/Under Accuracy: {over25_ensemble_accuracy:.3f}")
print(f"   Over/Under ROC AUC: {over25_ensemble_roc:.3f}")

print(f"\n📈 Improvement over baseline:")
print(f"   Outcome: +{outcome_ensemble_accuracy - 0.33:.1%} (vs random)")
print(f"   Over/Under: +{over25_ensemble_accuracy - 0.50:.1%} (vs random)")

print("\n✅ Enhanced training completed successfully!") 