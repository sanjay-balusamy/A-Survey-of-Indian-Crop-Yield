import gradio as gr
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Load & prepare data ────────────────────────────────────────────────────────
df = pd.read_csv('/home/jae/Desktop/derivarable 2/Crop-Yield-Prediction-based-on-Indian-Agriculture/Crop Prediction dataset.csv')

df['Production'] = pd.to_numeric(df['Production'], errors='coerce')
df['Production'] = df['Production'].fillna(df['Production'].mean())
df['Area']       = pd.to_numeric(df['Area'], errors='coerce')

for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
    df[col] = df[col].astype(str).str.strip()

df['Yield'] = df['Production'] / df['Area'].replace(0, np.nan)
df = df.dropna(subset=['Yield'])
df = df[df['Yield'] < df['Yield'].quantile(0.99)]

state_districts = {}
for state in sorted(df['State_Name'].unique()):
    state_districts[state] = sorted(df[df['State_Name'] == state]['District_Name'].unique().tolist())

all_states  = sorted(df['State_Name'].unique().tolist())
all_seasons = sorted(df['Season'].str.strip().unique().tolist())
all_crops   = sorted(df['Crop'].str.strip().unique().tolist())
year_min    = int(df['Crop_Year'].min())
year_max    = int(df['Crop_Year'].max())

# ── Train both models ─────────────────────────────────────────────────────────
print("Training models (this may take ~30 seconds)...")
features = ['Crop_Year', 'Temperature', 'Humidity', 'Soil_Moisture', 'Area',
            'State_Name', 'District_Name', 'Season', 'Crop']

data_enc = pd.get_dummies(df[features + ['Yield']])
X = data_enc.drop('Yield', axis=1)
y = data_enc['Yield']
model_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_r2 = r2_score(y_test, dt_model.predict(X_test))
print(f"Decision Tree R2: {dt_r2:.4f}")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_r2 = r2_score(y_test, rf_model.predict(X_test))
print(f"Random Forest  R2: {rf_r2:.4f}")

# ── Prediction function ───────────────────────────────────────────────────────
def encode_input(state, district, year, season, crop, temperature, humidity, soil_moisture, area):
    input_df = pd.DataFrame({
        'Crop_Year':     [int(year)],
        'Temperature':   [int(temperature)],
        'Humidity':      [int(humidity)],
        'Soil_Moisture': [int(soil_moisture)],
        'Area':          [float(area)],
        'State_Name':    [state],
        'District_Name': [district],
        'Season':        [season],
        'Crop':          [crop],
    })
    enc = pd.get_dummies(input_df)
    for col in model_columns:
        if col not in enc.columns:
            enc[col] = 0
    return enc[model_columns]

def predict_yield(state, district, year, season, crop, temperature, humidity, soil_moisture, area):
    try:
        enc = encode_input(state, district, year, season, crop, temperature, humidity, soil_moisture, area)

        dt_yield = dt_model.predict(enc)[0]
        rf_yield = rf_model.predict(enc)[0]

        dt_prod = dt_yield * float(area)
        rf_prod = rf_yield * float(area)
        avg_yield = (dt_yield + rf_yield) / 2
        avg_prod  = (dt_prod  + rf_prod)  / 2

        diff_pct = abs(dt_yield - rf_yield) / max(avg_yield, 0.0001) * 100

        verdict = (
            "Both models agree" if diff_pct < 10
            else ("Moderate disagreement" if diff_pct < 30
                  else "High disagreement — use RF result")
        )

        html = f"""
<div class="results-wrap">
  <div class="results-header">
    <span class="results-title">Prediction Results</span>
    <span class="results-meta">{crop} &middot; {season} &middot; {district}, {state}</span>
  </div>

  <div class="model-cards">
    <div class="model-card dt-card">
      <div class="model-badge">Decision Tree</div>
      <div class="model-r2">R&sup2; = {dt_r2:.4f}</div>
      <div class="metric-row">
        <span class="metric-label">Yield</span>
        <span class="metric-value">{dt_yield:.4f} <span class="metric-unit">t/ha</span></span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Production</span>
        <span class="metric-value">{dt_prod:,.1f} <span class="metric-unit">tonnes</span></span>
      </div>
    </div>

    <div class="model-card rf-card">
      <div class="model-badge">Random Forest</div>
      <div class="model-r2">R&sup2; = {rf_r2:.4f}</div>
      <div class="metric-row">
        <span class="metric-label">Yield</span>
        <span class="metric-value">{rf_yield:.4f} <span class="metric-unit">t/ha</span></span>
      </div>
      <div class="metric-row">
        <span class="metric-label">Production</span>
        <span class="metric-value">{rf_prod:,.1f} <span class="metric-unit">tonnes</span></span>
      </div>
    </div>
  </div>

  <div class="consensus-box">
    <div class="consensus-label">Ensemble Average</div>
    <div class="consensus-values">
      <div class="consensus-item">
        <div class="c-val">{avg_yield:.4f}</div>
        <div class="c-sub">tonnes / hectare</div>
      </div>
      <div class="consensus-divider"></div>
      <div class="consensus-item">
        <div class="c-val">{avg_prod:,.1f}</div>
        <div class="c-sub">total tonnes</div>
      </div>
      <div class="consensus-divider"></div>
      <div class="consensus-item">
        <div class="c-val">{float(area):,.0f}</div>
        <div class="c-sub">hectares</div>
      </div>
    </div>
    <div class="verdict">{verdict} &nbsp;&middot;&nbsp; {diff_pct:.1f}% difference</div>
  </div>

  <div class="note">Trained on 49,499 records across India (1997&ndash;2014)</div>
</div>
"""
        return html

    except Exception as e:
        return f'<div class="error-box"><b>Error:</b> {str(e)}</div>'


def update_districts(state):
    districts = state_districts.get(state, [])
    return gr.update(choices=districts, value=districts[0] if districts else None)


# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0d1117 !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #e6edf3 !important;
}

.hero {
    background: linear-gradient(135deg, #0d2818 0%, #0a3d1a 40%, #0f2d12 100%);
    border: 1px solid #1a4a22;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 10px;
}
.hero h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 400 !important;
    color: #f0fdf4 !important;
    margin: 0 0 10px !important;
    line-height: 1.15 !important;
}
.hero h1 em { font-style: italic; color: #86efac; }
.hero p {
    color: #86efac;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
    opacity: 0.85;
}

.section-heading {
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #4ade80 !important;
    margin: 20px 0 12px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #1a4a22 !important;
}

.gradio-container label {
    color: #a3c7a8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
}
input[type=number], select, .gr-input, .gr-dropdown, textarea {
    background: #161b22 !important;
    border: 1px solid #21361e !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
input[type=range] { accent-color: #4ade80 !important; }

.predict-btn button, button.predict-btn {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(21,128,61,0.35) !important;
    width: 100% !important;
}
.predict-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(21,128,61,0.5) !important;
    background: linear-gradient(135deg, #22c55e, #16a34a) !important;
}

.info-panel {
    background: #0d1f10;
    border: 1px solid #1a4a22;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}
.info-panel .info-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 14px;
}
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #1a3d1a;
    font-size: 0.83rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: #6b9e72; font-weight: 500; }
.info-val { color: #d1fae5; font-weight: 600; font-variant-numeric: tabular-nums; }

.results-wrap { font-family: 'DM Sans', sans-serif; color: #e6edf3; }
.results-header { display: flex; flex-direction: column; gap: 4px; margin-bottom: 20px; }
.results-title { font-size: 1.1rem; font-weight: 600; color: #f0fdf4; }
.results-meta { font-size: 0.8rem; color: #6b9e72; font-weight: 400; }

.model-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px; }
.model-card { border-radius: 12px; padding: 18px; position: relative; overflow: hidden; }
.dt-card { background: linear-gradient(145deg, #0a2918, #0d3520); border: 1px solid #1a5c2a; }
.rf-card { background: linear-gradient(145deg, #0a1f2e, #0d2a3d); border: 1px solid #1a3d5c; }

.model-badge { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 4px; }
.dt-card .model-badge { color: #4ade80; }
.rf-card .model-badge { color: #60a5fa; }
.model-r2 { font-size: 0.72rem; margin-bottom: 14px; opacity: 0.6; font-variant-numeric: tabular-nums; }

.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.metric-row:last-child { border-bottom: none; }
.metric-label { font-size: 0.78rem; color: #9ca3af; }
.metric-value { font-size: 1.05rem; font-weight: 700; font-variant-numeric: tabular-nums; color: #f0fdf4; }
.metric-unit { font-size: 0.7rem; font-weight: 400; color: #6b7280; margin-left: 2px; }

.consensus-box {
    background: linear-gradient(135deg, #0f2d1a, #0a2318);
    border: 1px solid #22543d;
    border-radius: 12px;
    padding: 20px 22px 16px;
    margin-bottom: 12px;
}
.consensus-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 14px;
}
.consensus-values {
    display: flex;
    align-items: center;
    justify-content: space-around;
    margin-bottom: 14px;
}
.consensus-item { text-align: center; }
.c-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    font-weight: 400;
    color: #86efac;
    font-variant-numeric: tabular-nums;
}
.c-sub { font-size: 0.7rem; color: #6b9e72; margin-top: 2px; }
.consensus-divider { width: 1px; height: 44px; background: #1a4a22; }
.verdict {
    font-size: 0.78rem;
    font-weight: 500;
    color: #a3c7a8;
    text-align: center;
    padding-top: 10px;
    border-top: 1px solid #1a4a22;
}
.note { font-size: 0.72rem; color: #4a6e50; text-align: right; font-style: italic; }
.error-box {
    background: #2d0a0a;
    border: 1px solid #7f1d1d;
    border-radius: 10px;
    padding: 16px;
    color: #fca5a5;
    font-size: 0.88rem;
}
.placeholder-box {
    background: #0d1f10;
    border: 1px dashed #1a4a22;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
}
.placeholder-icon { font-size: 3rem; margin-bottom: 12px; }
.placeholder-text { color: #4a6e50; font-size: 0.88rem; line-height: 1.6; }
.gr-panel, .gr-box { background: #0d1117 !important; border-color: #21361e !important; }
"""

PLACEHOLDER_HTML = """
<div class="placeholder-box">
  <div class="placeholder-icon">&#127807;</div>
  <div class="placeholder-text">
    Fill in the parameters on the left<br>
    and click <b>Predict Yield</b> to see<br>
    side-by-side model results.
  </div>
</div>
"""

# ── Gradio Layout ─────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Crop Yield Prediction — India",
    css=CSS,
    theme=gr.themes.Base(
        primary_hue="green",
        neutral_hue="slate",
    ),
) as demo:

    gr.HTML("""
    <div class="hero">
      <div class="hero-eyebrow">Machine Learning &middot; Agricultural Analytics</div>
      <h1>Crop <em>Yield</em> Prediction</h1>
      <p>Decision Tree &amp; Random Forest &mdash; side-by-side comparison across India (1997&ndash;2014)</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT: Inputs ──────────────────────────────────────────────────────
        with gr.Column(scale=4, elem_classes=["input-col"]):

            gr.HTML('<div class="section-heading">Location &amp; Time</div>')
            state_dd = gr.Dropdown(
                choices=all_states, value=all_states[0],
                label="State", interactive=True,
            )
            district_dd = gr.Dropdown(
                choices=state_districts[all_states[0]],
                value=state_districts[all_states[0]][0],
                label="District", interactive=True,
            )
            year_sl = gr.Slider(
                minimum=year_min, maximum=year_max + 10,
                value=2014, step=1,
                label="Crop Year",
            )

            gr.HTML('<div class="section-heading">Crop Details</div>')
            with gr.Row():
                season_dd = gr.Dropdown(
                    choices=all_seasons, value=all_seasons[0],
                    label="Season", interactive=True,
                )
                crop_dd = gr.Dropdown(
                    choices=all_crops, value="Paddy",
                    label="Crop", interactive=True,
                )
            area_num = gr.Number(
                label="Area (hectares)", value=500,
                minimum=1, maximum=500000,
            )

            gr.HTML('<div class="section-heading">Environmental Conditions</div>')
            temp_sl     = gr.Slider(0, 50,  value=28, step=1, label="Temperature (°C)")
            humidity_sl = gr.Slider(0, 100, value=70, step=1, label="Humidity (%)")
            soil_sl     = gr.Slider(0, 100, value=50, step=1, label="Soil Moisture (%)")

            predict_btn = gr.Button(
                "Predict Yield",
                variant="primary",
                size="lg",
                elem_classes=["predict-btn"],
            )

            gr.HTML(f"""
            <div class="info-panel">
              <div class="info-title">Model Info</div>
              <div class="info-row">
                <span class="info-key">Decision Tree R&sup2;</span>
                <span class="info-val">{dt_r2:.4f}</span>
              </div>
              <div class="info-row">
                <span class="info-key">Random Forest R&sup2;</span>
                <span class="info-val">{rf_r2:.4f}</span>
              </div>
              <div class="info-row">
                <span class="info-key">Training records</span>
                <span class="info-val">49,499</span>
              </div>
              <div class="info-row">
                <span class="info-key">RF Estimators</span>
                <span class="info-val">100 trees</span>
              </div>
              <div class="info-row">
                <span class="info-key">Test split</span>
                <span class="info-val">25%</span>
              </div>
            </div>
            """)

        # ── RIGHT: Results ────────────────────────────────────────────────────
        with gr.Column(scale=5):
            result_html = gr.HTML(value=PLACEHOLDER_HTML)

    # Wire up events
    state_dd.change(fn=update_districts, inputs=state_dd, outputs=district_dd)
    predict_btn.click(
        fn=predict_yield,
        inputs=[state_dd, district_dd, year_sl, season_dd, crop_dd,
                temp_sl, humidity_sl, soil_sl, area_num],
        outputs=result_html,
    )

if __name__ == "__main__":
    print("Launching Gradio app...")
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        
    )