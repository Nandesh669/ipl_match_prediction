import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import json

# Setup directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Datasets URLs
URLS = [
    "https://raw.githubusercontent.com/genieincodebottle/aiml-companion/main/projects/ipl-match-predictor/data/raw/matches.csv",
    "https://raw.githubusercontent.com/srinathkr07/IPL-Data-Analysis/master/matches.csv",
    "https://raw.githubusercontent.com/akashgupta4891/datasharing/master/matches.csv"
]

RAW_PATH = "data/raw/matches.csv"

def download_data():
    """Download matches.csv from available sources."""
    print("Attempting to download IPL Matches dataset...")
    for url in URLS:
        try:
            print(f"Trying URL: {url}")
            urllib.request.urlretrieve(url, RAW_PATH)
            print("Download successful!")
            return True
        except Exception as e:
            print(f"Failed to download from {url}. Error: {e}")
    return False

# Download the data
if not os.path.exists(RAW_PATH):
    success = download_data()
    if not success:
        print("Error: Could not download IPL Matches dataset. Please check your internet connection.")
        exit(1)
else:
    print("Matches dataset already exists locally.")

# Load the dataset
try:
    df = pd.read_csv(RAW_PATH)
    print(f"Dataset loaded. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Standardize column names to lowercase for consistency
df.columns = [col.lower() for col in df.columns]

# Standardize Team Names
team_mapping = {
    'delhi daredevils': 'Delhi Capitals',
    'delhi capitals': 'Delhi Capitals',
    'kings xi punjab': 'Punjab Kings',
    'punjab kings': 'Punjab Kings',
    'royal challengers bangalore': 'Royal Challengers Bengaluru',
    'royal challengers bengaluru': 'Royal Challengers Bengaluru',
    'deccan chargers': 'Sunrisers Hyderabad',
    'sunrisers hyderabad': 'Sunrisers Hyderabad',
    'rising pune supergiants': 'Rising Pune Supergiant',
    'rising pune supergiant': 'Rising Pune Supergiant',
    'pune warriors': 'Pune Warriors',
    'kochi tuskers kerala': 'Kochi Tuskers Kerala',
    'chennai super kings': 'Chennai Super Kings',
    'mumbai indians': 'Mumbai Indians',
    'rajasthan royals': 'Rajasthan Royals',
    'kolkata knight riders': 'Kolkata Knight Riders',
    'lucknow super giants': 'Lucknow Super Giants',
    'gujarat titans': 'Gujarat Titans',
    'gujarat lions': 'Gujarat Lions'
}

def standardize_team(team_name):
    if pd.isna(team_name):
        return np.nan
    val = str(team_name).strip().lower()
    return team_mapping.get(val, str(team_name).strip())

df['team1'] = df['team1'].apply(standardize_team)
df['team2'] = df['team2'].apply(standardize_team)
df['toss_winner'] = df['toss_winner'].apply(standardize_team)
df['winner'] = df['winner'].apply(standardize_team)

# Fill missing values for 'city' based on venue
venue_city_map = {
    'rajiv gandhi international stadium': 'Hyderabad',
    'rajiv gandhi international stadium, uppal': 'Hyderabad',
    'm chinnaswamy stadium': 'Bengaluru',
    'm. chinnaswamy stadium': 'Bengaluru',
    'wankhede stadium': 'Mumbai',
    'wankhede stadium, mumbai': 'Mumbai',
    'eden gardens': 'Kolkata',
    'eden gardens, kolkata': 'Kolkata',
    'feroz shah kotla': 'Delhi',
    'feroz shah kotla ground': 'Delhi',
    'arun jaitley stadium': 'Delhi',
    'arun jaitley stadium, delhi': 'Delhi',
    'ma chidambaram stadium': 'Chennai',
    'ma chidambaram stadium, chepauk': 'Chennai',
    'ma chidambaram stadium, chepauk, chennai': 'Chennai',
    'punjab cricket association IS Bindra Stadium, Mohali': 'Mohali',
    'punjab cricket association IS Bindra Stadium': 'Mohali',
    'punjab cricket association stadium, mohali': 'Mohali',
    'sawai mansingh stadium': 'Jaipur',
    'sawai mansingh stadium, jaipur': 'Jaipur',
    'narendra modi stadium': 'Ahmedabad',
    'narendra modi stadium, motera': 'Ahmedabad',
    'sardar patel stadium, motera': 'Ahmedabad',
}

def resolve_city(row):
    if not pd.isna(row['city']):
        return str(row['city']).strip()
    venue = str(row['venue']).lower().strip()
    for v_key, city in venue_city_map.items():
        if v_key in venue:
            return city
    return 'Unknown'

if 'city' in df.columns:
    df['city'] = df.apply(resolve_city, axis=1)

# Clean up rows where winner or team1/team2 is missing
df = df.dropna(subset=['team1', 'team2', 'winner'])
df = df[(df['winner'] == df['team1']) | (df['winner'] == df['team2'])]

# Save cleaned dataset for Tableau
tableau_df = df.copy()
tableau_df.to_csv("data/processed/tableau_matches_clean.csv", index=False)

# Build Team Statistics dataset for Tableau
all_teams = pd.concat([df['team1'], df['team2']]).unique()
team_stats = []
for team in all_teams:
    t_df = df[(df['team1'] == team) | (df['team2'] == team)]
    total_matches = len(t_df)
    wins = len(df[df['winner'] == team])
    toss_wins = len(df[df['toss_winner'] == team])
    toss_and_match_wins = len(df[(df['toss_winner'] == team) & (df['winner'] == team)])
    team_stats.append({
        'team': team,
        'total_matches': total_matches,
        'wins': wins,
        'win_percentage': round((wins / total_matches) * 100, 2) if total_matches > 0 else 0,
        'toss_wins': toss_wins,
        'toss_win_percentage': round((toss_wins / total_matches) * 100, 2) if total_matches > 0 else 0,
        'toss_and_match_wins': toss_and_match_wins,
        'toss_win_match_win_percentage': round((toss_and_match_wins / toss_wins) * 100, 2) if toss_wins > 0 else 0
    })
pd.DataFrame(team_stats).to_csv("data/processed/tableau_team_stats.csv", index=False)

# Build Venue Statistics dataset for Tableau
venue_stats = []
for venue in df['venue'].unique():
    v_df = df[df['venue'] == venue]
    total_v_matches = len(v_df)
    if total_v_matches < 5:
        continue
    
    bat_first_wins = 0
    field_first_wins = 0
    if 'toss_decision' in v_df.columns:
        bat_first_wins = len(v_df[
            ((v_df['toss_winner'] == v_df['winner']) & (v_df['toss_decision'] == 'bat')) |
            ((v_df['toss_winner'] != v_df['winner']) & (v_df['toss_decision'] == 'field'))
        ])
        field_first_wins = total_v_matches - bat_first_wins

    venue_stats.append({
        'venue': venue,
        'city': v_df['city'].iloc[0] if 'city' in v_df.columns else 'Unknown',
        'matches_played': total_v_matches,
        'bat_first_wins': bat_first_wins,
        'field_first_wins': field_first_wins,
        'bat_first_win_percentage': round((bat_first_wins / total_v_matches) * 100, 2) if total_v_matches > 0 else 0,
        'field_first_win_percentage': round((field_first_wins / total_v_matches) * 100, 2) if total_v_matches > 0 else 0
    })
pd.DataFrame(venue_stats).to_csv("data/processed/tableau_venue_stats.csv", index=False)
print("Saved clean datasets for Tableau visualization in data/processed/")

# Calculate Overall (Dataset-wide) Win Statistics for ML and Lookups
team_wins_dict = df['winner'].value_counts().to_dict()
team_matches_dict = (df['team1'].value_counts() + df['team2'].value_counts()).to_dict()

# Calculate overall team win ratios
team_win_ratio_dict = {}
for team in all_teams:
    w = team_wins_dict.get(team, 0)
    m = team_matches_dict.get(team, 1)
    team_win_ratio_dict[team] = float(w / m)

# Calculate overall head-to-head statistics
h2h_matches = {}
h2h_wins = {}
for idx, row in df.iterrows():
    t1, t2 = row['team1'], row['team2']
    winner = row['winner']
    pair = tuple(sorted([t1, t2]))
    pair_str = f"{pair[0]}|{pair[1]}"
    
    h2h_matches[pair_str] = h2h_matches.get(pair_str, 0) + 1
    h2h_wins[f"{winner}|{t1 if winner == t2 else t2}"] = h2h_wins.get(f"{winner}|{t1 if winner == t2 else t2}", 0) + 1

# Calculate overall venue statistics for teams
venue_matches = {}
venue_wins = {}
for idx, row in df.iterrows():
    t1, t2 = row['team1'], row['team2']
    venue = row['venue']
    winner = row['winner']
    
    venue_matches[f"{t1}|{venue}"] = venue_matches.get(f"{t1}|{venue}", 0) + 1
    venue_matches[f"{t2}|{venue}"] = venue_matches.get(f"{t2}|{venue}", 0) + 1
    
    venue_wins[f"{winner}|{venue}"] = venue_wins.get(f"{winner}|{venue}", 0) + 1

# Add columns using overall statistics
t1_win_ratios = []
t2_win_ratios = []
h2h_win_ratios = []
t1_venue_win_ratios = []
t2_venue_win_ratios = []

for idx, row in df.iterrows():
    t1 = row['team1']
    t2 = row['team2']
    venue = row['venue']
    
    # Team overall ratio
    t1_win_ratios.append(team_win_ratio_dict.get(t1, 0.5))
    t2_win_ratios.append(team_win_ratio_dict.get(t2, 0.5))
    
    # Head-to-Head ratio (Team 1 wins against Team 2)
    pair = tuple(sorted([t1, t2]))
    pair_str = f"{pair[0]}|{pair[1]}"
    m_h2h = h2h_matches.get(pair_str, 1)
    w_h2h = h2h_wins.get(f"{t1}|{t2}", 0)
    h2h_win_ratios.append(w_h2h / m_h2h)
    
    # Venue ratio
    t1_v_m = venue_matches.get(f"{t1}|{venue}", 1)
    t1_v_w = venue_wins.get(f"{t1}|{venue}", 0)
    t1_venue_win_ratios.append(t1_v_w / t1_v_m)
    
    t2_v_m = venue_matches.get(f"{t2}|{venue}", 1)
    t2_v_w = venue_wins.get(f"{t2}|{venue}", 0)
    t2_venue_win_ratios.append(t2_v_w / t2_v_m)

df['t1_win_ratio'] = t1_win_ratios
df['t2_win_ratio'] = t2_win_ratios
df['h2h_win_ratio'] = h2h_win_ratios
df['t1_venue_win_ratio'] = t1_venue_win_ratios
df['t2_venue_win_ratio'] = t2_venue_win_ratios

# Toss Winner mapping relative to Team 1
df['toss_winner_is_t1'] = (df['toss_winner'] == df['team1']).astype(int)
df['toss_decision_enc'] = (df['toss_decision'] == 'bat').astype(int)  # 1 for bat, 0 for field

# Select features
features_cols = ['t1_win_ratio', 't2_win_ratio', 'h2h_win_ratio', 't1_venue_win_ratio', 't2_venue_win_ratio', 'toss_winner_is_t1', 'toss_decision_enc']
df['y'] = (df['winner'] == df['team1']).astype(int)

# Create clean ML DataFrame
ml_df = df[features_cols + ['y', 'team1', 'team2', 'toss_winner', 'toss_decision', 'venue']].copy()

# Split into train/test sets based on original matches to avoid data leakage
train_idx, test_idx = train_test_split(ml_df.index, test_size=0.2, random_state=42, stratify=ml_df['y'])

train_df = ml_df.loc[train_idx].copy()
test_df = ml_df.loc[test_idx].copy()

# Double training set symmetrically
train_a = train_df[features_cols + ['y']].copy()
train_b = pd.DataFrame({
    't1_win_ratio': train_df['t2_win_ratio'],
    't2_win_ratio': train_df['t1_win_ratio'],
    'h2h_win_ratio': 1.0 - train_df['h2h_win_ratio'],
    't1_venue_win_ratio': train_df['t2_venue_win_ratio'],
    't2_venue_win_ratio': train_df['t1_venue_win_ratio'],
    'toss_winner_is_t1': 1 - train_df['toss_winner_is_t1'],
    'toss_decision_enc': train_df['toss_decision_enc'],
    'y': 1 - train_df['y']
})
X_train = pd.concat([train_a[features_cols], train_b[features_cols]], ignore_index=True)
y_train = pd.concat([train_a['y'], train_b['y']], ignore_index=True)

# Double test set symmetrically
test_a = test_df[features_cols + ['y']].copy()
test_b = pd.DataFrame({
    't1_win_ratio': test_df['t2_win_ratio'],
    't2_win_ratio': test_df['t1_win_ratio'],
    'h2h_win_ratio': 1.0 - test_df['h2h_win_ratio'],
    't1_venue_win_ratio': test_df['t2_venue_win_ratio'],
    't2_venue_win_ratio': test_df['t1_venue_win_ratio'],
    'toss_winner_is_t1': 1 - test_df['toss_winner_is_t1'],
    'toss_decision_enc': test_df['toss_decision_enc'],
    'y': 1 - test_df['y']
})
X_test = pd.concat([test_a[features_cols], test_b[features_cols]], ignore_index=True)
y_test = pd.concat([test_a['y'], test_b['y']], ignore_index=True)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# Models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
}

model_results = {}
best_acc = 0.0
best_model_name = None
best_model_obj = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("-" * 30)
    
    model_results[name] = {
        'accuracy': acc,
        'roc_auc': roc_auc,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'confusion_matrix': conf_mat
    }
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model_obj = model

print(f"\nBest Model: {best_model_name} with Accuracy: {best_acc:.4f}")

# Extract feature importance if available
feature_importances = {}
if hasattr(best_model_obj, 'feature_importances_'):
    importances = best_model_obj.feature_importances_
    for f, imp in zip(features_cols, importances):
        feature_importances[f] = float(imp)
elif hasattr(best_model_obj, 'coef_'):
    importances = np.abs(best_model_obj.coef_[0])
    norm_importances = importances / np.sum(importances)
    for f, imp in zip(features_cols, norm_importances):
        feature_importances[f] = float(imp)

# Save the best model and metadata
model_meta = {
    'best_model_name': best_model_name,
    'teams': sorted(list(all_teams)),
    'venues': sorted(list(df['venue'].unique())),
    'model_results': model_results,
    'feature_importances': feature_importances,
    'team_wins': team_wins_dict,
    'team_matches': team_matches_dict,
    'team_win_ratios': team_win_ratio_dict,
    'h2h_wins': h2h_wins,
    'h2h_matches': h2h_matches,
    'venue_wins': venue_wins,
    'venue_matches': venue_matches
}

with open("models/model_metadata.json", "w") as f:
    json.dump(model_meta, f, indent=4)

# Save best model
joblib.dump(best_model_obj, "models/best_model.pkl")

print("All models trained and metadata/best model serialized successfully!")
