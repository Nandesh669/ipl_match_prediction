import os
import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model and metadata
MODEL_PATH = "models/best_model.pkl"
META_PATH = "models/model_metadata.json"

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    print("Warning: Model or Metadata files do not exist. Please run train_model.py first.")
    model = None
    metadata = {}
else:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)

# Helper function to get team names and venues for dropdowns
def get_dropdown_data():
    if not metadata:
        return [], []
    return metadata.get("teams", []), metadata.get("venues", [])

@app.route("/")
def home():
    teams, venues = get_dropdown_data()
    return render_template("index.html", teams=teams, venues=venues, active_page="predict")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", active_page="dashboard")

@app.route("/model")
def model_page():
    # Pass metadata info to the page
    results = metadata.get("model_results", {})
    best_model = metadata.get("best_model_name", "Logistic Regression")
    importances = metadata.get("feature_importances", {})
    return render_template("model_info.html", results=results, best_model=best_model, importances=importances, active_page="model")

@app.route("/api/predict", methods=["POST"])
def predict():
    if not model or not metadata:
        return jsonify({"error": "Model is not loaded. Please train the model first."}), 500
    
    try:
        data = request.get_json()
        t1 = data.get("team1")
        t2 = data.get("team2")
        venue = data.get("venue")
        toss_winner = data.get("toss_winner")
        toss_decision = data.get("toss_decision") # 'bat' or 'field'
        
        if not all([t1, t2, venue, toss_winner, toss_decision]):
            return jsonify({"error": "Missing required fields"}), 400
        
        # Verify teams are different
        if t1 == t2:
            return jsonify({"error": "Team 1 and Team 2 must be different"}), 400

        # Retrieve lookup dictionaries from metadata
        team_win_ratios = metadata.get("team_win_ratios", {})
        h2h_matches = metadata.get("h2h_matches", {})
        h2h_wins = metadata.get("h2h_wins", {})
        venue_matches = metadata.get("venue_matches", {})
        venue_wins = metadata.get("venue_wins", {})
        
        # Compute features
        t1_ratio = team_win_ratios.get(t1, 0.5)
        t2_ratio = team_win_ratios.get(t2, 0.5)
        
        # Head to Head
        pair = tuple(sorted([t1, t2]))
        pair_str = f"{pair[0]}|{pair[1]}"
        m_h2h = h2h_matches.get(pair_str, 0)
        # wins of t1 against t2
        w_h2h = h2h_wins.get(f"{t1}|{t2}", 0)
        h2h_ratio = w_h2h / m_h2h if m_h2h > 0 else 0.5
        
        # Venue win ratios
        t1_v_m = venue_matches.get(f"{t1}|{venue}", 0)
        t1_v_w = venue_wins.get(f"{t1}|{venue}", 0)
        t1_v_ratio = t1_v_w / t1_v_m if t1_v_m > 0 else 0.5
        
        t2_v_m = venue_matches.get(f"{t2}|{venue}", 0)
        t2_v_w = venue_wins.get(f"{t2}|{venue}", 0)
        t2_v_ratio = t2_v_w / t2_v_m if t2_v_m > 0 else 0.5
        
        toss_winner_is_t1 = 1 if toss_winner == t1 else 0
        toss_decision_enc = 1 if toss_decision == "bat" else 0
        
        # Features array corresponding to features_cols
        # features_cols = ['t1_win_ratio', 't2_win_ratio', 'h2h_win_ratio', 't1_venue_win_ratio', 't2_venue_win_ratio', 'toss_winner_is_t1', 'toss_decision_enc']
        features = [[t1_ratio, t2_ratio, h2h_ratio, t1_v_ratio, t2_v_ratio, toss_winner_is_t1, toss_decision_enc]]
        
        # Predict probability of team 1 winning
        prob_t1 = float(model.predict_proba(features)[0][1])
        prob_t2 = 1.0 - prob_t1
        
        predicted_winner = t1 if prob_t1 >= prob_t2 else t2
        confidence = prob_t1 if prob_t1 >= prob_t2 else prob_t2
        
        # Additional statistics to return to front-end for visual enrichment
        team_matches_lookup = metadata.get("team_matches", {})
        team_wins_lookup = metadata.get("team_wins", {})
        
        response = {
            "predicted_winner": predicted_winner,
            "confidence": round(confidence * 100, 1),
            "team1": t1,
            "team2": t2,
            "team1_prob": round(prob_t1 * 100, 1),
            "team2_prob": round(prob_t2 * 100, 1),
            "team1_stats": {
                "matches": team_matches_lookup.get(t1, 0),
                "wins": team_wins_lookup.get(t1, 0),
                "win_rate": round(t1_ratio * 100, 1)
            },
            "team2_stats": {
                "matches": team_matches_lookup.get(t2, 0),
                "wins": team_wins_lookup.get(t2, 0),
                "win_rate": round(t2_ratio * 100, 1)
            },
            "h2h_stats": {
                "total": m_h2h,
                "team1_wins": w_h2h,
                "team2_wins": h2h_wins.get(f"{t2}|{t1}", 0)
            },
            "venue_stats": {
                "venue": venue,
                "team1_matches": t1_v_m,
                "team1_wins": t1_v_w,
                "team2_matches": t2_v_m,
                "team2_wins": t2_v_w
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def get_stats():
    """Retrieve processed stats CSV files and return as JSON for charts."""
    try:
        team_stats_path = "data/processed/tableau_team_stats.csv"
        venue_stats_path = "data/processed/tableau_venue_stats.csv"
        matches_clean_path = "data/processed/tableau_matches_clean.csv"
        
        team_df = pd.read_csv(team_stats_path)
        venue_df = pd.read_csv(venue_stats_path)
        matches_df = pd.read_csv(matches_clean_path)
        
        # Sort and limit for cleaner visualizations
        team_stats_list = team_df.sort_values(by="win_percentage", ascending=False).to_dict(orient="records")
        venue_stats_list = venue_df.sort_values(by="matches_played", ascending=False).head(15).to_dict(orient="records")
        
        # Calculate toss winner vs match winner overall statistics
        toss_match_same = len(matches_df[matches_df["toss_winner"] == matches_df["winner"]])
        toss_match_diff = len(matches_df) - toss_match_same
        toss_win_impact = {
            "toss_winner_won": toss_match_same,
            "toss_winner_lost": toss_match_diff,
            "percentage_toss_win_match_win": round((toss_match_same / len(matches_df)) * 100, 2)
        }
        
        # Calculate toss decision counts over seasons (field vs bat)
        toss_decisions = []
        if "season" in matches_df.columns and "toss_decision" in matches_df.columns:
            t_grouped = matches_df.groupby(["season", "toss_decision"]).size().unstack(fill_value=0)
            t_grouped = t_grouped.reset_index()
            toss_decisions = t_grouped.to_dict(orient="records")
            
        # Top 10 Player of the Match awards
        top_players = []
        if "player_of_match" in matches_df.columns:
            top_players = matches_df["player_of_match"].value_counts().head(10).reset_index().rename(columns={"index": "player", "count": "awards", "player_of_match": "player"}).to_dict(orient="records")

        return jsonify({
            "team_stats": team_stats_list,
            "venue_stats": venue_stats_list,
            "toss_win_impact": toss_win_impact,
            "toss_decisions": toss_decisions,
            "top_players": top_players
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
