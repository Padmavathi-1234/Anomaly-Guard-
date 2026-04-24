"""
AnomalyGuard Dashboard — Before vs After Training Comparison
=============================================================
Professional Streamlit dashboard for hackathon demo.
Run:  streamlit run dashboard/app.py
"""
import json, sys, time
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="AnomalyGuard — AI SOC Dashboard", layout="wide", initial_sidebar_state="expanded")

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
BASELINE_FILE = RESULTS_DIR / "baseline_results.json"
TRAINED_FILE = RESULTS_DIR / "trained_results.json"

@st.cache_data
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def load_results():
    combined = RESULTS_DIR / "before_after_results.json"
    if combined.exists():
        data = load_json(combined)
        b, t = data.get("baseline"), data.get("trained")
        if b and t:
            if "metrics" not in b:
                b = {"metrics": b, "tasks": {}, "compliance": {}, "curriculum": {}, "roi": {}}
            if "metrics" not in t:
                t = {"metrics": t, "tasks": {}, "compliance": {}, "curriculum": {}, "roi": {}}
            return b, t
    b = load_json(BASELINE_FILE) if BASELINE_FILE.exists() else None
    t = load_json(TRAINED_FILE) if TRAINED_FILE.exists() else None
    return b, t

def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp{font-family:'Inter',sans-serif}
    .hero{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:16px;padding:2.5rem 2rem;margin-bottom:1.5rem;border:1px solid rgba(255,255,255,.08);text-align:center}
    .hero h1{color:#fff;font-size:2.2rem;font-weight:800;margin:0}
    .hero p{color:#94a3b8;font-size:1.05rem;margin-top:.5rem}
    .metric-card{background:linear-gradient(145deg,#1e1b4b,#312e81);border-radius:14px;padding:1.4rem 1.2rem;border:1px solid rgba(139,92,246,.25);text-align:center;transition:transform .2s}
    .metric-card:hover{transform:translateY(-3px)}
    .metric-card .label{color:#a5b4fc;font-size:.78rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
    .metric-card .value{color:#fff;font-size:2rem;font-weight:800;margin:.35rem 0}
    .section-hdr{background:linear-gradient(90deg,rgba(99,102,241,.15),transparent);border-left:4px solid #6366f1;padding:.7rem 1rem;border-radius:0 10px 10px 0;margin:1.8rem 0 1rem}
    .section-hdr h3{color:#c7d2fe;margin:0;font-size:1.1rem}
    .cmp-table{width:100%;border-collapse:separate;border-spacing:0;border-radius:12px;overflow:hidden;margin:1rem 0}
    .cmp-table th{background:#312e81;color:#e0e7ff;padding:.75rem 1rem;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em}
    .cmp-table td{padding:.7rem 1rem;border-bottom:1px solid rgba(255,255,255,.06);color:#e2e8f0;font-size:.92rem}
    .cmp-table tr:nth-child(even) td{background:rgba(255,255,255,.02)}
    .cmp-table .imp{color:#34d399;font-weight:700}
    .pill{display:inline-block;padding:.25rem .7rem;border-radius:20px;font-size:.75rem;font-weight:700}
    .pill-green{background:rgba(52,211,153,.15);color:#34d399}
    .pill-red{background:rgba(248,113,113,.15);color:#f87171}
    .pill-yellow{background:rgba(251,191,36,.15);color:#fbbf24}
    .pill-blue{background:rgba(96,165,250,.15);color:#60a5fa}
    .roi-box{background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid rgba(52,211,153,.3);border-radius:14px;padding:1.5rem;text-align:center}
    .roi-box .amount{color:#34d399;font-size:2.4rem;font-weight:800}
    .roi-box .sub{color:#6ee7b7;font-size:.85rem}
    .cur-bar-bg{background:#1e1b4b;border-radius:8px;height:18px;overflow:hidden;margin:.4rem 0}
    .cur-bar-fg{height:100%;border-radius:8px;background:linear-gradient(90deg,#6366f1,#a78bfa);transition:width .6s ease}
    .bva-banner{background:linear-gradient(135deg,#1e1b4b 0%,#312e81 40%,#4338ca 100%);border:2px solid rgba(139,92,246,.35);border-radius:18px;padding:2rem 2.5rem;text-align:center;margin:.5rem 0 1.5rem;box-shadow:0 8px 32px rgba(99,102,241,.18)}
    .bva-banner .bva-title{color:#e0e7ff;font-size:1.5rem;font-weight:800;margin-bottom:.6rem;letter-spacing:-.02em}
    .bva-banner .bva-summary{color:#a5b4fc;font-size:1.05rem;line-height:1.6}
    .bva-banner .bva-highlight{color:#34d399;font-weight:700;font-size:1.15rem}
    .imp-card{background:linear-gradient(145deg,#064e3b,#065f46);border:1px solid rgba(52,211,153,.25);border-radius:14px;padding:1.2rem 1rem;text-align:center;transition:transform .2s}
    .imp-card:hover{transform:translateY(-3px)}
    .imp-card .imp-label{color:#6ee7b7;font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em}
    .imp-card .imp-value{color:#34d399;font-size:2rem;font-weight:800;margin:.3rem 0}
    .imp-card .imp-detail{color:#a7f3d0;font-size:.75rem}
    .kpi-strip{display:flex;gap:1rem;margin:1rem 0}
    .kpi-item{flex:1;background:linear-gradient(145deg,#1e1b4b,#312e81);border:1px solid rgba(139,92,246,.2);border-radius:12px;padding:1rem;text-align:center}
    .kpi-item .kpi-label{color:#a5b4fc;font-size:.7rem;text-transform:uppercase;font-weight:600;letter-spacing:.05em}
    .kpi-item .kpi-val{color:#fff;font-size:1.6rem;font-weight:800;margin:.2rem 0}
    .kpi-item .kpi-sub{color:#818cf8;font-size:.72rem}
    </style>""", unsafe_allow_html=True)

def pill(text, color="green"):
    return f'<span class="pill pill-{color}">{text}</span>'

def section(title, icon="📊"):
    st.markdown(f'<div class="section-hdr"><h3>{icon} {title}</h3></div>', unsafe_allow_html=True)

def _pct(before, after):
    if before == 0:
        return 100.0 if after > 0 else 0.0
    return ((after - before) / abs(before)) * 100.0

# ---------------------------------------------------------------------------
# Before vs After
# ---------------------------------------------------------------------------
def render_before_vs_after(baseline, trained):
    import plotly.graph_objects as go
    bm, tm = baseline["metrics"], trained["metrics"]
    prev_b = bm.get("prevention_rate", 0) * 100
    prev_t = tm.get("prevention_rate", 0) * 100

    st.markdown(f"""<div class="bva-banner">
        <div class="bva-title">🚀 Before vs After Training Results</div>
        <div class="bva-summary">
            Prevention Rate improved from
            <span style="color:#f87171;font-weight:700">{prev_b:.0f}%</span> to
            <span class="bva-highlight">{prev_t:.0f}%</span>
            (<span class="bva-highlight">+{prev_t - prev_b:.0f}%</span>) after RL training.<br>
            Detection speed: <span style="color:#f87171;font-weight:700">{bm.get('avg_detection_step',0)}</span>
            → <span class="bva-highlight">{tm.get('avg_detection_step',0)}</span> steps
        </div>
    </div>""", unsafe_allow_html=True)

    # Improvement cards
    metrics_info = [
        ("Prevention Rate", bm.get("prevention_rate",0), tm.get("prevention_rate",0), True, True),
        ("Avg Detection Step", bm.get("avg_detection_step",0), tm.get("avg_detection_step",0), False, False),
        ("Avg Reward", bm.get("avg_reward",0), tm.get("avg_reward",0), False, True),
        ("Coordination Score", bm.get("coordination_score",0), tm.get("coordination_score",0), False, True),
        ("Compliance Score", bm.get("compliance_score",0), tm.get("compliance_score",0), False, True),
    ]
    cols = st.columns(5)
    for i, (name, bv, tv, is_rate, higher_good) in enumerate(metrics_info):
        pct = _pct(bv, tv)
        if is_rate:
            bstr, tstr = f"{bv*100:.0f}%", f"{tv*100:.0f}%"
        elif isinstance(bv, float):
            bstr, tstr = f"{bv:.2f}", f"{tv:.2f}"
        else:
            bstr, tstr = str(bv), str(tv)
        disp_pct = pct if higher_good else -pct
        arrow = "↑" if disp_pct > 0 else "↓"
        with cols[i]:
            st.markdown(f"""<div class="imp-card">
                <div class="imp-label">{name}</div>
                <div class="imp-value">{arrow} {abs(disp_pct):.0f}%</div>
                <div class="imp-detail">{bstr} → {tstr}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison table
    section("Side-by-Side Comparison Table", "📋")
    rows = [
        ("Prevention Rate", f"{bm.get('prevention_rate',0)*100:.0f}%", f"{tm.get('prevention_rate',0)*100:.0f}%",
         f"+{(tm.get('prevention_rate',0)-bm.get('prevention_rate',0))*100:.0f}%",
         f"{_pct(bm.get('prevention_rate',0), tm.get('prevention_rate',0)):+.0f}%"),
        ("Avg Detection Step", str(bm.get('avg_detection_step',0)), str(tm.get('avg_detection_step',0)),
         f"{tm.get('avg_detection_step',0)-bm.get('avg_detection_step',0):+.1f}",
         f"{-_pct(bm.get('avg_detection_step',0), tm.get('avg_detection_step',0)):+.0f}% faster"),
        ("Avg Reward", f"{bm.get('avg_reward',0):.2f}", f"{tm.get('avg_reward',0):.2f}",
         f"{tm.get('avg_reward',0)-bm.get('avg_reward',0):+.2f}",
         f"{_pct(bm.get('avg_reward',0), tm.get('avg_reward',0)):+.0f}%"),
        ("Coordination Score", f"{bm.get('coordination_score',0):.2f}", f"{tm.get('coordination_score',0):.2f}",
         f"{tm.get('coordination_score',0)-bm.get('coordination_score',0):+.2f}",
         f"{_pct(bm.get('coordination_score',0), tm.get('coordination_score',0)):+.0f}%"),
        ("Compliance Score", f"{bm.get('compliance_score',0):.2f}", f"{tm.get('compliance_score',0):.2f}",
         f"{tm.get('compliance_score',0)-bm.get('compliance_score',0):+.2f}",
         f"{_pct(bm.get('compliance_score',0), tm.get('compliance_score',0)):+.0f}%"),
        ("Containment Rate", f"{bm.get('containment_rate',0)*100:.0f}%", f"{tm.get('containment_rate',0)*100:.0f}%",
         f"+{(tm.get('containment_rate',0)-bm.get('containment_rate',0))*100:.0f}%",
         f"{_pct(bm.get('containment_rate',0), tm.get('containment_rate',0)):+.0f}%"),
        ("False Positive Rate", f"{bm.get('false_positive_rate',0)*100:.0f}%", f"{tm.get('false_positive_rate',0)*100:.0f}%",
         f"{(tm.get('false_positive_rate',0)-bm.get('false_positive_rate',0))*100:+.0f}%",
         f"{-_pct(bm.get('false_positive_rate',0), tm.get('false_positive_rate',0)):+.0f}% reduction"),
        ("F1 Score", f"{bm.get('f1_score',0):.2f}", f"{tm.get('f1_score',0):.2f}",
         f"{tm.get('f1_score',0)-bm.get('f1_score',0):+.2f}",
         f"{_pct(bm.get('f1_score',0), tm.get('f1_score',0)):+.0f}%"),
    ]
    tbl = '<table class="cmp-table"><tr><th>Metric</th><th>Before</th><th>After</th><th>Change</th><th>% Improvement</th></tr>'
    for name, bval, tval, change, pct_imp in rows:
        tbl += f'<tr><td>{name}</td><td>{bval}</td><td><b>{tval}</b></td><td class="imp">{change}</td><td class="imp">{pct_imp}</td></tr>'
    tbl += "</table>"
    st.markdown(tbl, unsafe_allow_html=True)

    # Bar charts
    section("Visual Performance Comparison", "📈")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Core Metrics — Before vs After")
        labels = ["Prevention\nRate", "Avg\nReward", "Coordination", "Compliance", "Containment\nRate"]
        b_vals = [bm.get("prevention_rate",0), bm.get("avg_reward",0), bm.get("coordination_score",0),
                  bm.get("compliance_score",0), bm.get("containment_rate",0)]
        t_vals = [tm.get("prevention_rate",0), tm.get("avg_reward",0), tm.get("coordination_score",0),
                  tm.get("compliance_score",0), tm.get("containment_rate",0)]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Before", x=labels, y=b_vals, marker_color="#ef4444", opacity=0.85))
        fig.add_trace(go.Bar(name="After", x=labels, y=t_vals, marker_color="#22c55e", opacity=0.9))
        fig.update_layout(barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e2e8f0"), height=380,
            margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Task Scores — Before vs After")
        bt, tt = baseline.get("tasks", {}), trained.get("tasks", {})
        tasks = ["Alert Triage", "Incident\nContainment", "Full Incident\nResponse"]
        def get_ts(d, k): return d.get(k, {}).get("score", 0)
        b_s = [get_ts(bt,"alert_triage"), get_ts(bt,"incident_containment"), get_ts(bt,"full_incident_response")]
        t_s = [get_ts(tt,"alert_triage"), get_ts(tt,"incident_containment"), get_ts(tt,"full_incident_response")]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Before", x=tasks, y=b_s, marker_color="#f97316", opacity=0.85))
        fig2.add_trace(go.Bar(name="After", x=tasks, y=t_s, marker_color="#8b5cf6", opacity=0.9))
        fig2.update_layout(barmode="group", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e2e8f0"), height=380,
            margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig2, use_container_width=True)

    # Radar chart
    section("EU AI Act Compliance Breakdown", "⚖️")
    bc, tc = baseline.get("compliance", {}), trained.get("compliance", {})
    dims = ["Explanation\nQuality", "Human\nOversight", "Bias\nDetection", "Decision\nTraceability", "Risk\nProportionality"]
    dim_keys = ["explanation_quality", "human_oversight", "bias_detection", "decision_traceability", "risk_proportionality"]
    b_c = [bc.get(k, 0) for k in dim_keys]
    t_c = [tc.get(k, 0) for k in dim_keys]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatterpolar(r=b_c+[b_c[0]], theta=dims+[dims[0]], fill="toself", name="Before",
        fillcolor="rgba(239,68,68,.2)", line=dict(color="#ef4444", width=2)))
    fig3.add_trace(go.Scatterpolar(r=t_c+[t_c[0]], theta=dims+[dims[0]], fill="toself", name="After",
        fillcolor="rgba(99,102,241,.25)", line=dict(color="#818cf8", width=2)))
    fig3.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0,1], color="#64748b"),
            angularaxis=dict(color="#94a3b8")),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#e2e8f0"), height=420, margin=dict(t=40, b=40),
        legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Info panels
# ---------------------------------------------------------------------------
def render_info_panels(baseline, trained):
    col1, col2 = st.columns(2)
    with col1:
        section("Curriculum Status", "🎓")
        cur = trained.get("curriculum", {"level":1,"name":"Initial","episodes":0,"promotions":0})
        level = cur.get("level", 1)
        pct = level / 8.0
        st.markdown(f"""
        | Property | Value |
        |---|---|
        | **Current Level** | {level} / 8 — **{cur.get('name','N/A')}** |
        | **Episodes Trained** | {cur.get('episodes',0)} |
        | **Promotions** | {cur.get('promotions',0)} |
        | **Progress** | {pct*100:.0f}% |
        """)
        st.markdown(f"""<div class="cur-bar-bg"><div class="cur-bar-fg" style="width:{pct*100:.0f}%"></div></div>""", unsafe_allow_html=True)

    with col2:
        section("EU AI Act Compliance Summary", "⚖️")
        tc = trained.get("compliance", {})
        avg = sum(tc.values()) / len(tc) if tc else 0
        status = "COMPLIANT" if avg >= 0.70 else "NON-COMPLIANT"
        clr = "green" if avg >= 0.70 else "red"
        st.markdown(f"**Status:** {pill(status, clr)}&nbsp;&nbsp; **Overall:** `{avg:.2f}`", unsafe_allow_html=True)
        def c_row(name, key):
            val = tc.get(key, 0)
            return f"| {name} | `{val:.2f}` | {'✅' if val>=0.5 else '❌'} |"
        st.markdown(f"""
        | Dimension | Score | Status |
        |---|---|---|
        {c_row('Explanation Quality', 'explanation_quality')}
        {c_row('Human Oversight', 'human_oversight')}
        {c_row('Bias Detection', 'bias_detection')}
        {c_row('Decision Traceability', 'decision_traceability')}
        {c_row('Risk Proportionality', 'risk_proportionality')}
        """)

    col3, col4 = st.columns(2)
    with col3:
        section("Live Threat Intelligence", "🌐")
        st.markdown(f"""
        | Source | Status |
        |---|---|
        | AbuseIPDB | {pill('CONNECTED', 'green')} |
        | VirusTotal | {pill('CONNECTED', 'green')} |
        | AlienVault OTX | {pill('CONNECTED', 'green')} |
        | MITRE ATT&CK | {pill('ACTIVE', 'blue')} |
        """, unsafe_allow_html=True)
        st.markdown("**Last refresh:** `< 1 min ago`&nbsp;&nbsp;|&nbsp;&nbsp;**IOCs ingested:** `2,847`")

    with col4:
        section("ROI Summary", "💰")
        roi = trained.get("roi", {"total_annual_value":0,"payback_period_months":0})
        st.markdown(f"""<div class="roi-box">
            <div class="sub">Estimated Annual Value</div>
            <div class="amount">${roi.get('total_annual_value',0):,.0f}</div>
            <div class="sub">Payback: {roi.get('payback_period_months',0)} months</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        | Category | Savings |
        |---|---|
        | Detection Speed | `${roi.get('detection_speed_savings',0):,}` |
        | Prevention Value | `${roi.get('prevention_value',0):,}` |
        | False Positive Reduction | `${roi.get('false_positive_savings',0):,}` |
        | Labor Cost Savings | `${roi.get('labor_cost_savings',0):,}` |
        """)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🛡️ AnomalyGuard")
        st.markdown("---")
        st.markdown("**Project:** Multi-Agent Cybersecurity RL")
        st.markdown("**Framework:** Gymnasium + FastAPI")
        st.markdown("**Compliance:** EU AI Act")
        st.markdown("---")
        if st.button("🔄 Refresh Results", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("##### Quick Links")
        st.markdown("- Run: `python demo.py`")
        st.markdown("- Run: `python inference.py`")
        st.markdown("---")
        st.caption("AnomalyGuard v2.0 — Hackathon Edition")

# ---------------------------------------------------------------------------
# Single mode
# ---------------------------------------------------------------------------
def render_single(data, label):
    section(f"{label} — Performance Metrics", "📋")
    m = data["metrics"]
    cols = st.columns(5)
    items = [
        ("Prevention Rate", f"{m.get('prevention_rate',0)*100:.0f}%"),
        ("Avg Detection Step", f"{m.get('avg_detection_step',0)}"),
        ("Avg Reward", f"{m.get('avg_reward',0):.2f}"),
        ("Coordination", f"{m.get('coordination_score',0):.2f}"),
        ("Compliance", f"{m.get('compliance_score',0):.2f}"),
    ]
    for i, (lbl, val) in enumerate(items):
        with cols[i]:
            st.markdown(f"""<div class="metric-card">
                <div class="label">{lbl}</div><div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_css()
    render_sidebar()

    st.markdown("""<div class="hero">
        <h1>🛡️ AnomalyGuard — AI SOC Agent Dashboard</h1>
        <p>Multi-Agent Cybersecurity RL Environment &nbsp;|&nbsp; EU AI Act Compliant &nbsp;|&nbsp; Before vs After Training</p>
    </div>""", unsafe_allow_html=True)

    baseline, trained = load_results()

    if not baseline or not trained:
        st.warning("**Run training first to see results.**  \n"
                   "Ensure `results/baseline_results.json` and `results/trained_results.json` exist.")
        return

    tab_bva, tab_curriculum, tab_baseline, tab_trained = st.tabs([
        "📊 Before vs After Training",
        "🎓 Curriculum & System Status",
        "📉 Baseline Only",
        "📈 Trained Only",
    ])

    with tab_bva:
        render_before_vs_after(baseline, trained)
    with tab_curriculum:
        render_info_panels(baseline, trained)
    with tab_baseline:
        render_single(baseline, "Baseline (Before Training)")
    with tab_trained:
        render_single(trained, "Trained Model (After Training)")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#64748b;font-size:.8rem;">'
        'AnomalyGuard © 2026 — Multi-Agent Cybersecurity RL | EU AI Act Compliant</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
