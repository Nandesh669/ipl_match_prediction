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

# Calculate target column for stratified splitting
df['y'] = (df['winner'] == df['team1']).astype(int)

# Split into train/test sets based on original matches to avoid data leakage
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['y'])

# Import ML pipeline functions
from ml_pipeline import (
    double_dataset_symmetrically,
    run_hyperparameter_tuning,
    evaluate_and_calibrate_pipeline
)

# Symmetrically double training and test sets
X_train, y_train = double_dataset_symmetrically(train_df)
X_test, y_test = double_dataset_symmetrically(test_df)

# Perform GridSearchCV hyperparameter optimization on model candidates
best_model_name, best_pipeline, all_grid_results, best_results = run_hyperparameter_tuning(X_train, y_train)

# Calibrate and evaluate the best model pipeline
calibrated_pipeline, model_meta = evaluate_and_calibrate_pipeline(
    best_model_name=best_model_name,
    pipeline=best_pipeline,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    all_grid_results=all_grid_results
)

# Save the calibrated pipeline and metadata
with open("models/model_metadata.json", "w") as f:
    json.dump(model_meta, f, indent=4)

joblib.dump(calibrated_pipeline, "models/best_model.pkl")

print("All models trained, calibrated, and serialized successfully!")

