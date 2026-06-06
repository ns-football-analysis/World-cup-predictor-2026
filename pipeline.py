import os
import random
import numpy as np
import scipy.stats as stats
import pandas as pd

# ====================================================================
# 1. SMART, SELF-HEALING FIXTURES DATABASE LOADER
# ====================================================================
def get_live_fixtures():
    """
    Loads the full 104-match dataset from fixtures.csv. Automatically standardizes
    variations in column headers and auto-corrects Excel single-column formatting anomalies.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(base_dir, "fixtures.csv")
    
    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_filename, encoding='latin-1')
            except Exception as e:
                return f"Parsing Error (Encoding): {str(e)}"
        except Exception as e:
            return f"Parsing Error: {str(e)}"
            
        if len(df.columns) == 1 and ',' in df.columns[0]:
            raw_col = df.columns[0]
            headers = [h.strip('" ').strip() for h in raw_col.split(',')]
            df_split = df[raw_col].astype(str).str.split(',', expand=True)
            
            if df_split.shape[1] == len(headers):
                df_split.columns = headers
                for col in df_split.columns:
                    df_split[col] = df_split[col].astype(str).str.strip('" ').str.strip()
                df = df_split
            else:
                try:
                    df = pd.read_csv(csv_filename, encoding='latin-1', quoting=3)
                    df.columns = [c.strip('" ').strip() for c in df.columns]
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].astype(str).str.strip('" ').str.strip()
                except Exception as e:
                    return f"Failed to auto-parse single-column CSV layout: {str(e)}"
        
        rename_dict = {}
        for col in df.columns:
            clean_col = str(col).strip().lower().replace("_", "").replace(" ", "")
            
            if clean_col in ['home', 'hometeam', 'team1', 'hometeamname']:
                rename_dict[col] = 'home'
            elif clean_col in ['away', 'awayteam', 'team2', 'awayteamname']:
                rename_dict[col] = 'away'
            elif clean_col in ['stage', 'tournamentstage', 'phase', 'round']:
                rename_dict[col] = 'stage'
            elif clean_col in ['group', 'pool']:
                rename_dict[col] = 'group'
            elif clean_col in ['matchno', 'matchnum', 'matchnumber', 'id']:
                rename_dict[col] = 'match_no'
                
        df = df.rename(columns=rename_dict)
        
        missing_cols = [c for c in ['home', 'away'] if c not in df.columns]
        if missing_cols:
            return f"Missing required columns: {missing_cols}. Your actual CSV headers are: {list(df.columns)}"
            
        try:
            name_alignment = {
                "USA": "United States",
                "Czechia": "Czech Republic",
                "Ivory Coast": "Côte d'Ivoire"
            }
            df['home'] = df['home'].replace(name_alignment)
            df['away'] = df['away'].replace(name_alignment)
            
            return df
        except Exception as e:
            return f"Data Normalization Error: {str(e)}"
            
    return "File Not Found"

# ====================================================================
# 2. DATA ENGINEERING INTERFACES
# ====================================================================
def fetch_live_weather(lat, lon, override_condition=None):
    if override_condition and override_condition != "Auto-Detect (API Baseline)":
        return {"temp_celsius": 31.0 if "Heat" in override_condition else 20.0, "condition": override_condition}
    return {"temp_celsius": 22.5, "condition": "Standard / Mild Conditions"}

def fetch_live_injuries(team_name):
    mock_injuries = {"Mexico": 1, "United States": 2, "England": 1, "Brazil": 0, "Argentina": 1}
    return mock_injuries.get(team_name, 0)

# ====================================================================
# 3. BASE RATINGS GRID
# ====================================================================
def get_base_team_stats():
    data = {
        "Team": ["Mexico", "South Africa", "South Korea", "Czech Republic", "Canada", 
                 "Bosnia and Herzegovina", "United States", "Paraguay", "Brazil", "Morocco",
                 "Haiti", "Scotland", "Australia", "Türkiye", "Qatar", "Switzerland",
                 "Germany", "Curaçao", "Netherlands", "Japan", "Argentina", "Algeria", 
                 "England", "Croatia", "Portugal", "Congo DR", "Spain", "Cabo Verde", 
                 "Saudi Arabia", "Uruguay", "Belgium", "Egypt", "IR Iran", "New Zealand", 
                 "France", "Senegal", "Iraq", "Norway", "Austria", "Jordan", "Uzbekistan", 
                 "Colombia", "Ghana", "Panama", "Côte d'Ivoire", "Ecuador", "Sweden", "Tunisia"],
        "Base_Attack": [1.45, 1.05, 1.30, 1.25, 1.35, 1.10, 1.50, 1.00, 1.95, 1.40, 0.90, 1.20, 1.25, 1.35, 1.10, 1.45, 1.70, 0.85, 1.75, 1.40, 2.00, 1.15, 1.85, 1.50, 1.80, 1.05, 1.80, 0.95, 1.10, 1.65, 1.75, 1.20, 1.15, 0.90, 1.90, 1.35, 1.05, 1.40, 1.30, 1.00, 1.15, 1.55, 1.25, 1.10, 1.40, 1.35, 1.45, 1.10],
        "Base_Defense": [1.10, 1.30, 1.15, 1.20, 1.25, 1.35, 1.05, 1.15, 0.85, 0.95, 1.50, 1.20, 1.15, 1.10, 1.30, 1.00, 0.95, 1.60, 0.90, 1.05, 0.80, 1.25, 0.90, 1.00, 0.95, 1.25, 0.90, 1.30, 1.25, 0.95, 0.95, 1.15, 1.10, 1.40, 0.85, 1.05, 1.30, 1.10, 1.10, 1.35, 1.15, 1.00, 1.20, 1.25, 1.05, 1.10, 1.05, 1.15]
    }
    return pd.DataFrame(data).set_index("Team")

# ====================================================================
# 4. BIVARIATE DIXON-COLES MATH ENGINE (STAGE-AWARE)
# ====================================================================
def predict_match(home_team, away_team, weather_data, home_rest=5, away_rest=5, stage="Group Stage"):
    """
    Executes a Bivariate Dixon-Coles Poisson simulation to predict precise score lines,
    accounting for low-score dependence parameters. Resolves knockout ties to conclusion.
    """
    stats_df = get_base_team_stats()
    
    try:
        home_att = stats_df.loc[home_team, "Base_Attack"]
        home_def = stats_df.loc[home_team, "Base_Defense"]
    except KeyError:
        home_att, home_def = 1.30, 1.15
        
    try:
        away_att = stats_df.loc[away_team, "Base_Attack"]
        away_def = stats_df.loc[away_team, "Base_Defense"]
    except KeyError:
        away_att, away_def = 1.30, 1.15

    # Environmental Adjustments
    cond = weather_data["condition"]
    if "Heat" in cond:
        home_att *= 0.95; away_att *= 0.92  
    elif "Altitude" in cond:
        home_def *= 1.08; away_def *= 1.12  

    # Rest Modifiers
    if home_rest < 4: home_att *= 0.88; home_def *= 1.12
    if away_rest < 4: away_att *= 0.88; away_def *= 1.12

    # Injury Adjustments
    home_att *= (1.0 - (fetch_live_injuries(home_team) * 0.04))
    away_att *= (1.0 - (fetch_live_injuries(away_team) * 0.04))

    # Calculate Intensities
    avg_tournament_goals = 1.32
    home_lambda = home_att * away_def * avg_tournament_goals
    away_lambda = away_att * home_def * avg_tournament_goals

    # 1. Generate Base Independent Distributions
    max_goals = 8
    home_poisson = stats.poisson.pmf(range(max_goals), home_lambda)
    away_poisson = stats.poisson.pmf(range(max_goals), away_lambda)
    matrix = np.outer(home_poisson, away_poisson)

    # 2. Inject Dixon-Coles Low-Score Dependence Tuning (tau matrix adjustments)
    rho = -0.06  # Standard empirical scale parameter for tournament distributions
    if home_lambda > 0 and away_lambda > 0:
        matrix[0, 0] *= (1.0 - (home_lambda * away_lambda * rho))
        matrix[0, 1] *= (1.0 + (home_lambda * rho))
        matrix[1, 0] *= (1.0 + (away_lambda * rho))
        matrix[1, 1] *= (1.0 - rho)
        
        # Guard rail: prevent mathematical anomalies crossing sub-zero thresholds
        matrix = np.clip(matrix, 0, None)

    # 3. Re-normalize Distribution Density
    matrix_sum = np.sum(matrix)
    if matrix_sum > 0:
        matrix = matrix / matrix_sum

    # Calculate Outcomes
    home_win_prob = float(np.sum(np.tril(matrix, -1)))
    draw_prob = float(np.sum(np.diag(matrix)))
    away_win_prob = float(np.sum(np.triu(matrix, 1)))
    
    # Identify highest probability score coordinates
    raw_home_score = int(matrix.argmax() // max_goals)
    raw_away_score = int(matrix.argmax() % max_goals)
    expected_score_str = f"{raw_home_score} - {raw_away_score}"

    output = {
        "home_win": home_win_prob,
        "draw": draw_prob,
        "away_win": away_win_prob,
        "expected_score": expected_score_str,
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "extra_time_played": False,
        "penalties_triggered": False,
        "team_advancing": None,
        "live_metrics": {"temperature": weather_data["temp_celsius"], "condition": cond}
    }

    # ====================================================================
    # KNOCKOUT STAGE COLD RESOLUTIONS
    # ====================================================================
    is_knockout = str(stage).strip().lower() != "group stage"

    if is_knockout:
        outcomes = ["home", "draw", "away"]
        weights = [home_win_prob, draw_prob, away_win_prob]
        simulated_outcome = random.choices(outcomes, weights=weights)[0]

        if simulated_outcome == "home":
            output["team_advancing"] = home_team
        elif simulated_outcome == "away":
            output["team_advancing"] = away_team
        else:
            # Draw triggered -> extra time simulation (30 minutes)
            output["extra_time_played"] = True
            et_home_lambda = home_lambda / 3.0
            et_away_lambda = away_lambda / 3.0
            
            et_home_goals = np.random.poisson(et_home_lambda)
            et_away_goals = np.random.poisson(et_away_lambda)
            
            final_home_score = raw_home_score + et_home_goals
            final_away_score = raw_away_score + et_away_goals
            output["expected_score"] = f"{final_home_score} - {final_away_score}"
            
            if et_home_goals > et_away_goals:
                output["team_advancing"] = home_team
                output["expected_score"] += " (AET)"
            elif et_away_goals > et_home_goals:
                output["team_advancing"] = away_team
                output["expected_score"] += " (AET)"
            else:
                # Still level -> Resolve via Penalty Shootout
                output["penalties_triggered"] = True
                pk_winner = random.choice([home_team, away_team])
                output["team_advancing"] = pk_winner
                output["expected_score"] += " (Pens)"
    else:
        output["team_advancing"] = "N/A (Group Stage Draw)"

    return output