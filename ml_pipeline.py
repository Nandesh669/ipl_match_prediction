import logging
import time
from functools import wraps
from typing import Dict, List, Tuple, Any, Union, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IPL_ML_Pipeline")

# Custom Exceptions
class PipelineError(Exception):
    """Base exception for the IPL ML Pipeline."""
    pass

class DataValidationError(PipelineError):
    """Exception raised when input data fails validation checks."""
    pass

# Custom Decorator for timing
def log_duration(func):
    """Decorator to log the execution duration of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Method '{func.__name__}' execution started.")
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"Method '{func.__name__}' completed successfully in {duration:.4f} seconds.")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Method '{func.__name__}' failed after {duration:.4f} seconds with error: {e}")
            raise
    return wrapper

class IPLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer to handle all feature engineering for IPL matches.
    It fits on historical training matches to learn win ratios, head-to-head stats, 
    and venue win rates, and maps raw inputs to numerical features at inference time.
    """
    def __init__(self) -> None:
        self.team_win_ratios_: Dict[str, float] = {}
        self.team_wins_: Dict[str, int] = {}
        self.team_matches_: Dict[str, int] = {}
        self.h2h_matches_: Dict[str, int] = {}
        self.h2h_wins_: Dict[str, int] = {}
        self.venue_matches_: Dict[str, int] = {}
        self.venue_wins_: Dict[str, int] = {}
        self.all_teams_: List[str] = []
        self.all_venues_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "IPLFeatureExtractor":
        """
        Fits lookups using training dataframe X and targets y (1 if team1 wins, 0 if team2 wins).
        """
        if y is None:
            raise DataValidationError("Target vector 'y' is required to fit IPLFeatureExtractor.")

        # Ensure correct column naming and structure
        X = X.copy()
        X.columns = [col.lower() for col in X.columns]
        required_cols = {"team1", "team2", "venue", "toss_winner", "toss_decision"}
        missing_cols = required_cols - set(X.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns for feature extraction: {missing_cols}")

        logger.info(f"Fitting feature extractor on {len(X)} samples...")

        # Reconstruct actual match winner name for internal statistical aggregation
        # If y is 1, winner is team1. If y is 0, winner is team2.
        winners = np.where(y == 1, X["team1"], X["team2"])

        # 1. Overall Team Win Ratios
        team1_counts = X["team1"].value_counts()
        team2_counts = X["team2"].value_counts()
        team_matches = team1_counts.add(team2_counts, fill_value=0)
        
        # Convert winners array to Series to count wins per team
        team_wins = pd.Series(winners).value_counts()

        self.all_teams_ = sorted(list(set(X["team1"].unique()).union(set(X["team2"].unique()))))
        self.all_venues_ = sorted(list(X["venue"].unique()))

        for team in self.all_teams_:
            m = int(team_matches.get(team, 0))
            w = int(team_wins.get(team, 0))
            self.team_matches_[team] = m
            self.team_wins_[team] = w
            self.team_win_ratios_[team] = float(w / m) if m > 0 else 0.5

        # Head to Head wins and matches
        self.h2h_matches_ = {}
        self.h2h_wins_ = {}
        for (_, row), w in zip(X.iterrows(), winners):
            t1, t2 = row["team1"], row["team2"]
            pair = tuple(sorted([t1, t2]))
            pair_str = f"{pair[0]}|{pair[1]}"
            self.h2h_matches_[pair_str] = self.h2h_matches_.get(pair_str, 0) + 1
            self.h2h_wins_[f"{w}|{t1 if w == t2 else t2}"] = self.h2h_wins_.get(f"{w}|{t1 if w == t2 else t2}", 0) + 1

        # Venue win ratios
        self.venue_matches_ = {}
        self.venue_wins_ = {}
        for (_, row), w in zip(X.iterrows(), winners):
            t1, t2 = row["team1"], row["team2"]
            venue = row["venue"]
            
            self.venue_matches_[f"{t1}|{venue}"] = self.venue_matches_.get(f"{t1}|{venue}", 0) + 1
            self.venue_matches_[f"{t2}|{venue}"] = self.venue_matches_.get(f"{t2}|{venue}", 0) + 1
            self.venue_wins_[f"{w}|{venue}"] = self.venue_wins_.get(f"{w}|{venue}", 0) + 1

        logger.info("Feature extractor fitted successfully.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw IPL matches columns to numerical features.
        """
        X = X.copy()
        X.columns = [col.lower() for col in X.columns]

        t1_win_ratios: List[float] = []
        t2_win_ratios: List[float] = []
        h2h_win_ratios: List[float] = []
        t1_venue_win_ratios: List[float] = []
        t2_venue_win_ratios: List[float] = []
        toss_winner_is_t1: List[int] = []
        toss_decision_enc: List[int] = []

        for _, row in X.iterrows():
            t1 = row["team1"]
            t2 = row["team2"]
            venue = row["venue"]
            toss_w = row["toss_winner"]
            toss_d = str(row["toss_decision"]).strip().lower()

            # Team ratios (default to 0.5 if team is unseen)
            t1_win_ratios.append(self.team_win_ratios_.get(t1, 0.5))
            t2_win_ratios.append(self.team_win_ratios_.get(t2, 0.5))

            # Head to Head ratios
            pair = tuple(sorted([t1, t2]))
            pair_str = f"{pair[0]}|{pair[1]}"
            m_h2h = self.h2h_matches_.get(pair_str, 0)
            w_h2h = self.h2h_wins_.get(f"{t1}|{t2}", 0)
            h2h_win_ratios.append(w_h2h / m_h2h if m_h2h > 0 else 0.5)

            # Venue win ratios
            t1_v_m = self.venue_matches_.get(f"{t1}|{venue}", 0)
            t1_v_w = self.venue_wins_.get(f"{t1}|{venue}", 0)
            t1_venue_win_ratios.append(t1_v_w / t1_v_m if t1_v_m > 0 else 0.5)

            t2_v_m = self.venue_matches_.get(f"{t2}|{venue}", 0)
            t2_v_w = self.venue_wins_.get(f"{t2}|{venue}", 0)
            t2_venue_win_ratios.append(t2_v_w / t2_v_m if t2_v_m > 0 else 0.5)

            # Toss features
            toss_winner_is_t1.append(1 if toss_w == t1 else 0)
            toss_decision_enc.append(1 if toss_d == "bat" else 0)

        return pd.DataFrame({
            "t1_win_ratio": t1_win_ratios,
            "t2_win_ratio": t2_win_ratios,
            "h2h_win_ratio": h2h_win_ratios,
            "t1_venue_win_ratio": t1_venue_win_ratios,
            "t2_venue_win_ratio": t2_venue_win_ratios,
            "toss_winner_is_t1": toss_winner_is_t1,
            "toss_decision_enc": toss_decision_enc
        }, index=X.index)


@log_duration
def double_dataset_symmetrically(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Symmetrically doubles the dataset by swapping team1 and team2.
    Ensures that the model learns position-independent features.
    """
    df_a = df.copy()
    
    # Swap team1 and team2 in df_b
    df_b = df.copy()
    df_b["team1"] = df["team2"]
    df_b["team2"] = df["team1"]
    
    # Concatenate df_a and df_b
    df_symmetric = pd.concat([df_a, df_b], ignore_index=True)
    
    # Calculate target (1 if winner is team1, else 0)
    y_symmetric = (df_symmetric["winner"] == df_symmetric["team1"]).astype(int)
    
    X_symmetric = df_symmetric[["team1", "team2", "venue", "toss_winner", "toss_decision"]]
    
    logger.info(f"Symmetrized dataset from {len(df)} rows to {len(X_symmetric)} rows.")
    return X_symmetric, y_symmetric


@log_duration
def run_hyperparameter_tuning(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Tuple[str, Pipeline, Dict[str, Any], Dict[str, Any]]:
    """
    Performs Grid Search on several candidate classifiers in a Pipeline
    to find the best model and parameters.
    """
    # Define models and grids
    # We prefix param names with classifier__ because they are passed to the classifier step in the Pipeline
    models_config = {
        "Logistic Regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__penalty": ["l2"]
            }
        },
        "Decision Tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "classifier__max_depth": [3, 5, 8, 12],
                "classifier__min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 6, 8, 10],
                "classifier__min_samples_leaf": [1, 2, 4]
            }
        },
        "Gradient Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [50, 100, 150],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.01, 0.05, 0.1]
            }
        }
    }

    best_overall_score = -1.0
    best_model_name = ""
    best_pipeline = None
    all_grid_results = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, config in models_config.items():
        logger.info(f"Tuning hyper-parameters for {name}...")
        
        # Build training pipeline
        pipe = Pipeline([
            ("extractor", IPLFeatureExtractor()),
            ("scaler", StandardScaler()),
            ("classifier", config["estimator"])
        ])

        grid_search = GridSearchCV(
            estimator=pipe,
            param_grid=config["param_grid"],
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        # Save results for UI display
        # Convert NumPy types in cv_results_ to native Python types for JSON compatibility
        cv_results = grid_search.cv_results_
        params_list = []
        for i, params in enumerate(cv_results["params"]):
            cleaned_params = {k.replace("classifier__", ""): v for k, v in params.items()}
            params_list.append({
                "params": cleaned_params,
                "mean_score": float(cv_results["mean_test_score"][i]),
                "std_score": float(cv_results["std_test_score"][i])
            })
            
        all_grid_results[name] = {
            "best_score": float(best_score),
            "best_params": {k.replace("classifier__", ""): v for k, v in best_params.items()},
            "candidates": params_list
        }

        logger.info(f"Model: {name} | Best CV Accuracy: {best_score:.4f} | Params: {best_params}")

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_model_name = name
            best_pipeline = grid_search.best_estimator_

    logger.info(f"Best overall model: {best_model_name} with CV accuracy: {best_overall_score:.4f}")
    return best_model_name, best_pipeline, all_grid_results, all_grid_results[best_model_name]


@log_duration
def evaluate_and_calibrate_pipeline(
    best_model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    all_grid_results: Dict[str, Any]
) -> Tuple[CalibratedClassifierCV, Dict[str, Any]]:
    """
    Calibrates the best pipeline classifier, computes Stratified K-Fold CV scores,
    and returns ROC, Precision-Recall, and Calibration curve coordinates for visualization.
    """
    logger.info(f"Calibrating final pipeline with CalibratedClassifierCV...")
    
    # We calibrate the pipeline.
    calibrated_pipeline = CalibratedClassifierCV(estimator=pipeline, cv=5, method="sigmoid")
    calibrated_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = calibrated_pipeline.predict(X_test)
    y_pred_proba = calibrated_pipeline.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_loss = log_loss(y_test, y_pred_proba)
    test_brier = brier_score_loss(y_test, y_pred_proba)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"Calibrated Test Metrics | Acc: {test_acc:.4f} | AUC: {test_auc:.4f} | LogLoss: {test_loss:.4f} | Brier: {test_brier:.4f}")

    # Generate ROC Curve coordinates
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    # Downsample curves to 30 points to reduce payload size while keeping curves smooth
    step_roc = max(1, len(fpr) // 30)
    roc_coordinates = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in range(0, len(fpr), step_roc)]
    # Ensure final point (1, 1) is included
    if roc_coordinates[-1] != {"fpr": 1.0, "tpr": 1.0}:
        roc_coordinates.append({"fpr": 1.0, "tpr": 1.0})

    # Generate Precision-Recall Curve coordinates
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    step_pr = max(1, len(precision) // 30)
    pr_coordinates = [{"precision": float(precision[i]), "recall": float(recall[i])} for i in range(0, len(precision), step_pr)]

    # Generate Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    calibration_coordinates = [
        {"true_prob": float(prob_true[i]), "pred_prob": float(prob_pred[i])} for i in range(len(prob_true))
    ]

    # Perform 5-fold CV to get fold scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    fold_accuracies = [float(score) for score in cv_results["test_score"]]

    # Extract feature importances if the underlying classifier supports it
    classifier = pipeline.named_steps["classifier"]
    extractor = pipeline.named_steps["extractor"]
    
    # Feature columns used by extractor
    feature_cols = ["t1_win_ratio", "t2_win_ratio", "h2h_win_ratio", "t1_venue_win_ratio", "t2_venue_win_ratio", "toss_winner_is_t1", "toss_decision_enc"]
    feature_importances = {}
    
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        feature_importances = {f: float(imp) for f, imp in zip(feature_cols, importances)}
    elif hasattr(classifier, "coef_"):
        coef = np.abs(classifier.coef_[0])
        norm_coef = coef / np.sum(coef)
        feature_importances = {f: float(c) for f, c in zip(feature_cols, norm_coef)}
    else:
        # Default to uniform values
        feature_importances = {f: 1.0 / len(feature_cols) for f in feature_cols}

    # Package model metadata
    model_metadata = {
        "best_model_name": best_model_name,
        "validation_accuracy": float(test_acc),
        "validation_auc": float(test_auc),
        "validation_log_loss": float(test_loss),
        "validation_brier_score": float(test_brier),
        "confusion_matrix": conf_matrix,
        "precision": float(report["weighted avg"]["precision"]),
        "recall": float(report["weighted avg"]["recall"]),
        "f1_score": float(report["weighted avg"]["f1-score"]),
        "fold_accuracies": fold_accuracies,
        "feature_importances": feature_importances,
        "roc_curve": roc_coordinates,
        "pr_curve": pr_coordinates,
        "calibration_curve": calibration_coordinates,
        "hyperparameters": all_grid_results,
        "teams": extractor.all_teams_,
        "venues": extractor.all_venues_,
        # Keep old mapping keys so other app parts don't break
        "team_wins": extractor.team_wins_,
        "team_matches": extractor.team_matches_,
        "team_win_ratios": extractor.team_win_ratios_,
        "h2h_wins": extractor.h2h_wins_,
        "h2h_matches": extractor.h2h_matches_,
        "venue_wins": extractor.venue_wins_,
        "venue_matches": extractor.venue_matches_
    }

    return calibrated_pipeline, model_metadata
