import streamlit as st
import pandas as pd
import plotly.express as px
from pipeline import get_live_fixtures, predict_match, fetch_live_weather

# Page configuration
st.set_page_config(page_title="2026 World Cup Predictor", layout="wide", page_icon="⚽")

st.title("🏆 2026 FIFA World Cup Predictive Engine")
st.markdown("World Cup Predictor based on Dixon Coles Model")
st.write("---")

# ====================================================================
# 1. LOAD AND CHECK FIXTURES DATA
# ====================================================================
data_payload = get_live_fixtures()

if isinstance(data_payload, str):
    if data_payload == "File Not Found":
        st.error("⚠️ **Missing Data File:** `fixtures.csv` was not found in your project folder. Displaying sample backup data below.")
    else:
        st.error(f"⚠️ **CSV Structure Error:** Could not parse file. Details: {data_payload}")
    
    df_fixtures = pd.DataFrame([
        {"match_no": 1, "date": "2026-06-11", "stage": "Group Stage", "group": "Group A", "home": "Mexico", "away": "South Africa", "venue": "Estadio Azteca"},
        {"match_no": 2, "date": "2026-06-11", "stage": "Group Stage", "group": "Group A", "home": "South Korea", "away": "Czech Republic", "venue": "Estadio Akron"},
        {"match_no": 3, "date": "2026-06-12", "stage": "Group Stage", "group": "Group B", "home": "Canada", "away": "Bosnia and Herzegovina", "venue": "BMO Field"}
    ])
else:
    df_fixtures = data_payload

if 'stage' in df_fixtures.columns and 'group' in df_fixtures.columns:
    df_fixtures['category'] = df_fixtures.apply(
        lambda r: r['group'] if str(r['stage']).strip() == 'Group Stage' else r['stage'], axis=1
    )
else:
    df_fixtures['category'] = 'Group Stage'

categories = df_fixtures['category'].unique()

# ====================================================================
# 2. DUAL DROP-DOWN LAYOUT
# ====================================================================
col1, col2 = st.columns(2)

with col1:
    selected_cat = st.selectbox("🗂️ Select Group or Tournament Stage", options=categories, index=0)

df_filtered = df_fixtures[df_fixtures['category'] == selected_cat].copy()

df_filtered['display_label'] = df_filtered.apply(
    lambda r: f"Match {r.get('match_no', 'N/A')}: {r['home']} vs {r['away']}", axis=1
)

with col2:
    selected_fixture_label = st.selectbox("⚽ Select Specific Fixture", options=df_filtered['display_label'].tolist())

selected_match = df_filtered[df_filtered['display_label'] == selected_fixture_label].iloc[0]

# ====================================================================
# 3. ENVIRONMENT & CONTROLS INTERFACE (SIDEBAR)
# ====================================================================
st.sidebar.header("🛠️ Simulation Modifiers")

weather_override = st.sidebar.selectbox(
    "Weather Condition Override",
    ["Auto-Detect (API Baseline)", "Standard / Mild Conditions", "Extreme Heat (30°C+)", "High Altitude Impact"]
)

home_rest = st.sidebar.slider(f"{selected_match['home']} Days of Rest", 2, 10, 5)
away_rest = st.sidebar.slider(f"{selected_match['away']} Days of Rest", 2, 10, 5)

# ====================================================================
# 4. ENGINE RUN & RESULTS DISPLAY
# ====================================================================
weather_data = fetch_live_weather(lat=40.0, lon=-90.0, override_condition=weather_override)

res = predict_match(
    home_team=selected_match['home'],
    away_team=selected_match['away'],
    weather_data=weather_data,
    home_rest=home_rest,
    away_rest=away_rest,
    stage=selected_match.get('stage', 'Group Stage')
)

# --- VISUAL ALIGNMENT ENGINE: FORCE PRECISE 100% SUM ---
p_home = round(res['home_win'] * 100, 1)
p_draw = round(res['draw'] * 100, 1)
p_away = round(res['away_win'] * 100, 1)

# Calculate the visual rounding remainder
remainder = round(100.0 - (p_home + p_draw + p_away), 1)

if remainder != 0.0:
    # Safely inject the micro-remainder into the highest probability outcome
    probabilities = [p_home, p_draw, p_away]
    highest_idx = probabilities.index(max(probabilities))
    if highest_idx == 0:
        p_home = round(p_home + remainder, 1)
    elif highest_idx == 1:
        p_draw = round(p_draw + remainder, 1)
    else:
        p_away = round(p_away + remainder, 1)

# Render Heading
st.write("---")
st.subheader(f"📊 Dixon-Coles Match Forecast: {selected_match['home']} vs {selected_match['away']}")

# Metrics Cards Layout (Guaranteed perfectly balanced sum display)
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric(label=f"🟢 {selected_match['home']} Win (90 Mins)", value=f"{p_home:.1f}%")
with m_col2:
    st.metric(label="⚪ Draw (90 Mins)", value=f"{p_draw:.1f}%")
with m_col3:
    st.metric(label=f"🔵 {selected_match['away']} Win (90 Mins)", value=f"{p_away:.1f}%")

# ====================================================================
# 5. KNOCKOUT OVERLAY VISUALIZATION
# ====================================================================
is_knockout_match = str(selected_match.get('stage', '')).strip().lower() != "group stage"

if is_knockout_match:
    st.write("")
    ko_col1, ko_col2 = st.columns([1, 2])
    
    with ko_col1:
        st.success(f"🏆 **Projected to Advance:** \n### {res['team_advancing']}")
    
    with ko_col2:
        if res['penalties_triggered']:
            st.warning("⚠️ **Deadlock Overtime Notice:** The match remained a tie through 120 minutes. Progression was resolved via a stochastically calculated Penalty Shootout.")
        elif res['extra_time_played']:
            st.warning("⏱️ **Extra Time Notice:** The match ended in a draw during regulatory 90 minutes. Deadlock was broken during simulated Extra Time.")
        else:
            st.info("🎯 **Clean Progression:** The tie was completely resolved within standard regulatory 90 minutes without needing extra time periods.")

st.write("")

# Dynamic Bar Chart Output (Using aligned variables)
chart_df = pd.DataFrame({
    'Match Outcome': [selected_match['home'], 'Draw', selected_match['away']],
    'Probability Weight': [p_home / 100, p_draw / 100, p_away / 100]
})
fig = px.bar(chart_df, x='Match Outcome', y='Probability Weight', text_auto='.1%',
             color='Match Outcome', color_discrete_sequence=["#2ca02c", "#bcbd22", "#1f77b4"])
fig.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# Footer Specs Notice Box
st.info(
    f"📋 **Data Specification Matrix:** \n"
    f"• Venue context: `{selected_match.get('venue', 'Standard Stadium')}` | Expected Date: `{selected_match.get('date', 'N/A')}` \n"
    f"• Projected Score Trend line: **{res['expected_score']}** \n"
    f"• Unmodified baseline tracking limits: {selected_match['home']} xG: `{res['home_lambda']:.2f}` | {selected_match['away']} xG: `{res['away_lambda']:.2f}`"
)