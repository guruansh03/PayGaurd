"""
app.py — PayGuard Real-time UPI Fraud Monitoring Dashboard
Themed version: Bloomberg Terminal / Broadsheet / Swiss Editorial
+ Toolbar hidden, chart entrance animations, enriched Overview, improved Live Score
FIX: sidebar theme buttons now visible in all 3 themes
"""

import os
import sys
import logging
import shutil
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    THRESHOLD, IF_THRESHOLD, AE_THRESHOLD, LOF_THRESHOLD,
    DATA_PATH, CHARTS_DIR, ANOMALY_SCORES_PATH,
    Y_TEST_PATH, X_TEST_PATH, X_VAL_PATH, SENDER_STATS_PATH,
    IF_MODEL_PATH, IF_SCALER_PATH,
    AE_MODEL_PATH, AE_SCALER_PATH, AE_MSE_SCALER_PATH,
    AE_HIDDEN, OUTPUTS_DIR, RANDOM_SEED,
    LOF_MODEL_PATH, LOF_SCALER_PATH,
)

log = logging.getLogger(__name__)


def _configure_tesseract_binary():
    """Return a usable Tesseract executable path when OCR support is available."""
    tesseract_cmd = os.getenv("TESSERACT_CMD")
    if tesseract_cmd and os.path.exists(tesseract_cmd):
        return tesseract_cmd

    detected = shutil.which("tesseract")
    if detected:
        return detected

    if os.name == "nt":
        candidates = [
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

    return None

import sqlite3, hashlib, json, threading
from datetime import datetime as _audit_dt

_audit_db_path = os.path.join(OUTPUTS_DIR, 'audit_log.db')
_audit_lock = threading.Lock()

def _ensure_audit_table():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with sqlite3.connect(_audit_db_path, timeout=5) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            page TEXT NOT NULL,
            model TEXT,
            threshold REAL,
            input_hash TEXT,
            n_rows INTEGER,
            n_flagged INTEGER,
            top_score REAL,
            verdict TEXT,
            details TEXT
        )''')
        conn.commit()

_ensure_audit_table()

def log_audit(page, model=None, threshold=None, input_data=None, n_rows=0, n_flagged=0, top_score=0.0, verdict='', details=''):
    try:
        input_hash = hashlib.md5(str(input_data).encode()).hexdigest()[:16] if input_data is not None else ''
        with _audit_lock:
            with sqlite3.connect(_audit_db_path, timeout=5) as conn:
                conn.execute(
                    'INSERT INTO audit_log (timestamp, page, model, threshold, input_hash, n_rows, n_flagged, top_score, verdict, details) VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (_audit_dt.utcnow().isoformat(), page, model, threshold, input_hash, n_rows, n_flagged, top_score, verdict, details)
                )
                conn.commit()
    except Exception as e:
        log.warning(f"Audit log write failed: {e}")

UPI_P2P_LIMIT      = 100_000
UPI_MERCHANT_LIMIT = 200_000
UPI_MAX_LIMIT      = 500_000

st.set_page_config(page_title='PayGuard', page_icon='🔷', layout='wide', initial_sidebar_state='expanded')

# ── Theme system ──
THEMES = {
    'bloomberg': {'label': 'Bloomberg Terminal', 'btn_bg': '#FF8C00', 'btn_color': '#07090D'},
    'broadsheet': {'label': 'Broadsheet',        'btn_bg': '#1A1A1A', 'btn_color': '#F4EFE4'},
    'swiss':      {'label': 'Swiss Editorial',   'btn_bg': '#E8320A', 'btn_color': '#FFF'},
}

if 'ui_theme' not in st.session_state:
    st.session_state['ui_theme'] = 'bloomberg'

_theme = st.session_state['ui_theme']

# ── Per-theme CSS ──
_THEME_CSS = {
'bloomberg': """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
@keyframes fadeInUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes bbTick { from{opacity:0;transform:translateX(-6px)} to{opacity:1;transform:translateX(0)} }
@keyframes bbGlow { 0%,100%{box-shadow:0 0 4px rgba(255,140,0,.2)} 50%{box-shadow:0 0 12px rgba(255,140,0,.5)} }
@keyframes chartRise { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }
@keyframes chartFadeScale { from{opacity:0;transform:scale(0.96)} to{opacity:1;transform:scale(1)} }
@keyframes chartSlideLeft { from{opacity:0;transform:translateX(-20px)} to{opacity:1;transform:translateX(0)} }

html, body, [class*="css"] { background:#0E131A !important; color:#C0C8D0 !important; font-family:'Courier New',Courier,monospace !important; font-size:14px !important; }
[data-testid="stAppViewContainer"],[data-testid="stMain"] { background:#0E131A !important; }
.main .block-container { background:#0E131A !important; max-width:1400px !important; padding:1.5rem 2rem !important; }
[data-testid="stSidebar"],[data-testid="stSidebar"] > div { background:#07090D !important; border-right:1px solid #1A2430 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,[data-testid="stSidebar"] label { color:#8AABB8 !important; font-size:12px !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 { color:#C0C8D0 !important; }
[data-testid="stMarkdown"] p,[data-testid="stMarkdownContainer"] p,[data-testid="stText"] { color:#C0C8D0 !important; font-family:'Courier New',monospace !important; }
[data-testid="stMarkdown"] h1,[data-testid="stMarkdown"] h2,[data-testid="stMarkdown"] h3,[data-testid="stMarkdown"] h4 { color:#C0C8D0 !important; font-family:'Courier New',monospace !important; }
[data-testid="metric-container"] { background:#07090D !important; border:1px solid #1A2430 !important; border-radius:0 !important; box-shadow:none !important; padding:16px !important; animation:bbTick .4s ease both; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'Courier New',monospace !important; color:#C0C8D0 !important; font-size:1.7rem !important; }
[data-testid="metric-container"] [data-testid="stMetricLabel"] p { color:#8AABB8 !important; letter-spacing:.14em; text-transform:uppercase; font-size:10px !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { color:#00C896 !important; }
.stTabs [data-baseweb="tab-list"] { background:#07090D !important; border-bottom:1px solid #1A2430; }
.stTabs [data-baseweb="tab"] { background:transparent !important; border:none !important; border-radius:0 !important; color:#4A6070 !important; font-size:11px !important; letter-spacing:.12em; text-transform:uppercase; padding:8px 18px !important; }
.stTabs [aria-selected="true"] { background:transparent !important; border-bottom:2px solid #FF8C00 !important; color:#FF8C00 !important; font-weight:700 !important; box-shadow:none !important; }
.stButton > button { background:transparent !important; border:1px solid #1A2430 !important; color:#8AABB8 !important; border-radius:0 !important; font-family:'Courier New',monospace !important; letter-spacing:.1em; font-size:12px !important; padding:6px 14px !important; transition:all .15s ease !important; }
.stButton > button:hover { border-color:#FF8C00 !important; color:#FF8C00 !important; background:rgba(255,140,0,.06) !important; }
.stButton > button[kind="primary"] { background:#FF8C00 !important; border:none !important; color:#07090D !important; font-weight:700 !important; }
.stButton > button[kind="primary"]:hover { background:#FFA030 !important; }
[data-testid="stSidebar"] .stButton > button { background:transparent !important; border:1px solid #2A3A48 !important; color:#8AABB8 !important; border-radius:0 !important; font-family:'Courier New',monospace !important; letter-spacing:.1em; font-size:12px !important; }
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button p { color:#8AABB8 !important; }
[data-testid="stSidebar"] .stButton > button:hover { border-color:#FF8C00 !important; color:#FF8C00 !important; background:rgba(255,140,0,.06) !important; }
[data-testid="stSidebar"] .stButton > button:hover span,
[data-testid="stSidebar"] .stButton > button:hover p { color:#FF8C00 !important; }
[data-testid="stDataFrame"] { border-radius:0 !important; border:1px solid #1A2430 !important; animation:chartFadeScale .4s ease both; animation-delay:.1s; }
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th { color:#C0C8D0 !important; background:#07090D !important; font-family:'Courier New',monospace !important; font-size:12px !important; }
.stSlider [data-testid="stMarkdownContainer"] p { color:#8AABB8 !important; font-size:11px !important; }
[data-testid="stSelectbox"] label p { color:#8AABB8 !important; }
[data-testid="stSelectbox"] > div > div { background:#07090D !important; border:1px solid #1A2430 !important; color:#C0C8D0 !important; border-radius:0 !important; }
.stExpander { border:1px solid #1A2430 !important; border-radius:0 !important; }
.stExpander summary p { color:#8AABB8 !important; }
[data-testid="stHorizontalRule"] { border-color:#1A2430 !important; }
[data-testid="stAlert"] p { color:#8AABB8 !important; }
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input { background:#07090D !important; border:1px solid #1A2430 !important; color:#C0C8D0 !important; border-radius:0 !important; font-family:'Courier New',monospace !important; }
[data-testid="stFileUploader"] { border:1px dashed #1A2430 !important; background:#07090D !important; border-radius:0 !important; }
[data-testid="stFileUploader"] p { color:#8AABB8 !important; }
[data-testid="stCaption"] p { color:#4A6070 !important; font-size:11px !important; }
.stRadio label p { color:#8AABB8 !important; }
.stCheckbox label p { color:#8AABB8 !important; }
[data-testid="stFileUploader"] section { padding:10px 12px !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] { min-height:84px !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] > div { padding:4px 0 !important; }
[data-testid="stFileUploader"] button { border-radius:0 !important; }
.ocr-panel { background:#07090D !important; border:1px solid #1A2430 !important; padding:16px 18px !important; margin:8px 0 12px 0 !important; }
.ocr-panel__title { color:#FF8C00 !important; font-family:'Courier New',monospace !important; font-size:12px !important; letter-spacing:.18em !important; text-transform:uppercase !important; margin:0 0 6px 0 !important; }
.ocr-panel__subtitle { color:#8AABB8 !important; font-family:'Courier New',monospace !important; font-size:12px !important; margin:0 !important; line-height:1.5 !important; }
.ocr-panel__meta { display:flex !important; flex-wrap:wrap !important; gap:6px !important; margin-top:10px !important; }
.ocr-chip { display:inline-flex !important; align-items:center !important; gap:6px !important; padding:3px 8px !important; border:1px solid #1A2430 !important; background:rgba(255,255,255,.02) !important; color:#C0C8D0 !important; font-family:'Courier New',monospace !important; font-size:10px !important; letter-spacing:.08em !important; text-transform:uppercase !important; }
.ocr-chip--accent { border-color:#FF8C00 !important; color:#FF8C00 !important; background:rgba(255,140,0,.08) !important; }
.ocr-chip--muted { color:#8AABB8 !important; }
.ocr-status { margin-top:10px !important; padding:10px 12px !important; border:1px solid #1A2430 !important; background:#0B1118 !important; color:#C0C8D0 !important; font-family:'Courier New',monospace !important; font-size:12px !important; line-height:1.45 !important; }
.ocr-status--success { border-color:#00C896 !important; background:rgba(0,200,150,.08) !important; color:#00C896 !important; }
.ocr-status--warning { border-color:#FF8C00 !important; background:rgba(255,140,0,.08) !important; color:#FFB347 !important; }
.ocr-status--info { border-color:#4A6070 !important; background:rgba(74,96,112,.12) !important; color:#8AABB8 !important; }
.ocr-results { display:flex !important; flex-wrap:wrap !important; gap:8px !important; margin-top:10px !important; }
.ocr-result { display:inline-flex !important; align-items:center !important; padding:4px 10px !important; border:1px solid #1A2430 !important; background:#07090D !important; color:#C0C8D0 !important; font-family:'Courier New',monospace !important; font-size:11px !important; }
.ocr-result--amount { border-color:#FF8C00 !important; color:#FF8C00 !important; }
.ocr-result--upi { border-color:#00C896 !important; color:#00C896 !important; }
[data-testid="stAlert"] { border-radius:0 !important; }
[data-testid="stPlotlyChart"] { animation:chartRise .55s cubic-bezier(0.16,1,0.3,1) both; }
[data-testid="stPlotlyChart"]:nth-child(1){animation-delay:.00s}
[data-testid="stPlotlyChart"]:nth-child(2){animation-delay:.12s}
[data-testid="stPlotlyChart"]:nth-child(3){animation-delay:.24s}
[data-testid="stPlotlyChart"]:nth-child(4){animation-delay:.36s}
[data-testid="stPlotlyChart"]:nth-child(5){animation-delay:.48s}
[data-testid="stPlotlyChart"]:nth-child(6){animation-delay:.60s}

.hero-banner { background:#07090D !important; border:1px solid #FF8C00 !important; border-radius:0 !important; padding:24px 28px !important; margin-bottom:16px !important; animation:bbGlow 3s ease infinite; }
.hero-banner h1 { color:#FF8C00 !important; font-family:'Courier New',monospace !important; font-size:1.7rem !important; letter-spacing:.12em !important; margin:0 0 6px 0 !important; }
.hero-banner p { color:#8AABB8 !important; font-family:'Courier New',monospace !important; font-size:.85rem !important; margin:0 !important; }
.hero-banner::before { display:none !important; }
.hero-badge { background:rgba(255,140,0,.1) !important; border:1px solid rgba(255,140,0,.3) !important; color:#FF8C00 !important; border-radius:0 !important; font-family:'Courier New',monospace !important; font-size:.7rem !important; letter-spacing:.14em; padding:2px 8px !important; }
.kpi-card { border-radius:0 !important; border-left-width:2px !important; animation:bbTick .4s ease both; }
.kpi-blue { background:#07090D !important; border-left-color:#4A6070 !important; }
.kpi-red   { background:#07090D !important; border-left-color:#E84040 !important; }
.kpi-green { background:#07090D !important; border-left-color:#00C896 !important; }
.kpi-amber { background:#07090D !important; border-left-color:#FF8C00 !important; }
.kpi-value { color:#C0C8D0 !important; font-family:'Courier New',monospace !important; font-size:1.5rem !important; }
.kpi-label { color:#8AABB8 !important; font-family:'Courier New',monospace !important; letter-spacing:.14em; font-size:10px !important; }
.section-header { color:#FF8C00 !important; letter-spacing:.18em !important; font-family:'Courier New',monospace !important; font-size:11px !important; text-transform:uppercase; animation:chartSlideLeft .4s ease both; }
.verdict-fraud { background:#150000 !important; border:1px solid #E84040 !important; border-radius:0 !important; box-shadow:0 0 20px rgba(232,64,64,.25) !important; padding:24px !important; animation:chartFadeScale .45s cubic-bezier(0.34,1.56,0.64,1) both; }
.verdict-fraud h2,.verdict-fraud p,.verdict-fraud strong { color:#FF6B6B !important; }
.verdict-normal { background:#001510 !important; border:1px solid #00C896 !important; border-radius:0 !important; box-shadow:0 0 20px rgba(0,200,150,.15) !important; padding:24px !important; animation:chartFadeScale .45s cubic-bezier(0.34,1.56,0.64,1) both; }
.verdict-normal h2,.verdict-normal p,.verdict-normal strong { color:#00C896 !important; }
.fraud-tag { background:rgba(232,64,64,.14) !important; border:1px solid rgba(232,64,64,.28) !important; color:#FF6B6B !important; border-radius:0 !important; font-family:'Courier New',monospace !important; font-size:11px !important; padding:2px 8px !important; margin:2px !important; display:inline-block !important; }
.stat-card { background:#07090D !important; border:1px solid #1A2430 !important; border-radius:0 !important; animation:fadeInUp .5s ease both; }
.stat-card h2 { font-family:'Courier New',monospace !important; background:none !important; -webkit-background-clip:unset !important; -webkit-text-fill-color:#FF8C00 !important; font-size:2rem !important; }
.stat-card p { color:#8AABB8 !important; font-size:12px !important; }
.pipeline-step { background:#07090D !important; border:1px solid #1A2430 !important; border-radius:0 !important; transition:border-color .2s,box-shadow .2s !important; }
.pipeline-step:hover { border-color:#FF8C00 !important; box-shadow:0 0 8px rgba(255,140,0,.15) !important; }
""",

'broadsheet': """
@import url('https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&display=swap');
@keyframes fadeInUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes slideIn { from{opacity:0;transform:translateX(-8px)} to{opacity:1;transform:translateX(0)} }
@keyframes chartRise { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }
@keyframes chartFadeScale { from{opacity:0;transform:scale(0.96)} to{opacity:1;transform:scale(1)} }
@keyframes chartSlideLeft { from{opacity:0;transform:translateX(-20px)} to{opacity:1;transform:translateX(0)} }

html, body, [class*="css"] { background:#F4EFE4 !important; color:#1A1A1A !important; font-family:'Times New Roman',Times,serif !important; font-size:15px !important; }
[data-testid="stAppViewContainer"],[data-testid="stMain"] { background:#F4EFE4 !important; }
.main .block-container { background:#F4EFE4 !important; max-width:1400px !important; padding:1.5rem 2rem !important; }
[data-testid="stSidebar"],[data-testid="stSidebar"] > div { background:#EDE8D8 !important; border-right:2px solid #1A1A1A !important; }
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,[data-testid="stSidebar"] label { color:#555 !important; font-family:'Courier New',monospace !important; font-size:12px !important; }
[data-testid="stMarkdown"] p,[data-testid="stMarkdownContainer"] p,[data-testid="stText"] { color:#1A1A1A !important; font-family:'Times New Roman',serif !important; font-size:14px !important; }
[data-testid="stMarkdown"] h1,[data-testid="stMarkdown"] h2,[data-testid="stMarkdown"] h3,[data-testid="stMarkdown"] h4 { color:#1A1A1A !important; }
[data-testid="metric-container"] { background:#F4EFE4 !important; border:1.5px solid #1A1A1A !important; border-radius:0 !important; box-shadow:none !important; padding:16px !important; animation:slideIn .4s ease both; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'Courier New',monospace !important; color:#1A1A1A !important; font-size:1.7rem !important; }
[data-testid="metric-container"] [data-testid="stMetricLabel"] p { color:#666 !important; letter-spacing:.14em; text-transform:uppercase; font-size:9px !important; font-family:'Courier New',monospace !important; }
.stTabs [data-baseweb="tab-list"] { background:#EDE8D8 !important; border-bottom:2px solid #1A1A1A; }
.stTabs [data-baseweb="tab"] { background:transparent !important; border:none !important; border-radius:0 !important; color:#888 !important; font-family:'Courier New',monospace !important; font-size:11px !important; letter-spacing:.1em; text-transform:uppercase; padding:8px 18px !important; }
.stTabs [aria-selected="true"] { background:#F4EFE4 !important; border-bottom:3px solid #1A1A1A !important; color:#1A1A1A !important; font-weight:700 !important; box-shadow:none !important; }
.stButton > button { background:#1A1A1A !important; border:none !important; color:#F4EFE4 !important; border-radius:0 !important; font-family:'Courier New',monospace !important; letter-spacing:.1em; font-size:12px !important; padding:6px 16px !important; transition:background .15s !important; }
.stButton > button:hover { background:#333 !important; color:#F4EFE4 !important; }
.stButton > button[kind="primary"] { background:#1A1A1A !important; color:#F4EFE4 !important; font-weight:700 !important; }
[data-testid="stSidebar"] .stButton > button { background:#1A1A1A !important; border:none !important; color:#F4EFE4 !important; border-radius:0 !important; font-family:'Courier New',monospace !important; letter-spacing:.1em; font-size:12px !important; padding:6px 16px !important; }
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button div { color:#F4EFE4 !important; }
[data-testid="stSidebar"] .stButton > button:hover { background:#333 !important; color:#F4EFE4 !important; }
[data-testid="stSidebar"] .stButton > button:hover span,
[data-testid="stSidebar"] .stButton > button:hover p { color:#F4EFE4 !important; }
[data-testid="stDataFrame"] { border-radius:0 !important; border:1.5px solid #1A1A1A !important; animation:chartFadeScale .4s ease both; animation-delay:.1s; }
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th { color:#1A1A1A !important; background:#F4EFE4 !important; font-family:'Courier New',monospace !important; font-size:12px !important; }
.stSlider [data-testid="stMarkdownContainer"] p { color:#555 !important; font-family:'Courier New',monospace !important; font-size:11px !important; }
[data-testid="stSelectbox"] label p { color:#555 !important; font-family:'Courier New',monospace !important; }
[data-testid="stSelectbox"] > div > div { background:#F4EFE4 !important; border:1.5px solid #1A1A1A !important; color:#1A1A1A !important; border-radius:0 !important; }
.stExpander { border:1.5px solid #1A1A1A !important; border-radius:0 !important; }
.stExpander summary p { color:#1A1A1A !important; font-family:'Courier New',monospace !important; }
[data-testid="stHorizontalRule"] { border-color:#C8C0A8 !important; }
[data-testid="stAlert"] p { color:#555 !important; }
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input { background:#F4EFE4 !important; border:1.5px solid #1A1A1A !important; color:#1A1A1A !important; border-radius:0 !important; font-family:'Courier New',monospace !important; }
[data-testid="stFileUploader"] { border:1.5px dashed #888 !important; background:#EDE8D8 !important; border-radius:0 !important; }
[data-testid="stFileUploader"] p { color:#555 !important; }
[data-testid="stCaption"] p { color:#888 !important; font-family:'Courier New',monospace !important; font-size:11px !important; }
.stRadio label p { color:#555 !important; }
.stCheckbox label p { color:#555 !important; }
[data-testid="stFileUploader"] section { padding:10px 12px !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] { min-height:84px !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] > div { padding:4px 0 !important; }
[data-testid="stFileUploader"] button { border-radius:0 !important; }
.ocr-panel { background:#F4EFE4 !important; border:1.5px solid #1A1A1A !important; padding:16px 18px !important; margin:8px 0 12px 0 !important; }
.ocr-panel__title { color:#1A1A1A !important; font-family:'Courier New',monospace !important; font-size:12px !important; letter-spacing:.18em !important; text-transform:uppercase !important; margin:0 0 6px 0 !important; }
.ocr-panel__subtitle { color:#555 !important; font-family:'Courier New',monospace !important; font-size:12px !important; margin:0 !important; line-height:1.5 !important; }
.ocr-panel__meta { display:flex !important; flex-wrap:wrap !important; gap:6px !important; margin-top:10px !important; }
.ocr-chip { display:inline-flex !important; align-items:center !important; gap:6px !important; padding:3px 8px !important; border:1px solid #1A1A1A !important; background:rgba(0,0,0,.02) !important; color:#1A1A1A !important; font-family:'Courier New',monospace !important; font-size:10px !important; letter-spacing:.08em !important; text-transform:uppercase !important; }
.ocr-chip--accent { border-color:#1A1A1A !important; color:#1A1A1A !important; background:rgba(0,0,0,.04) !important; }
.ocr-chip--muted { color:#666 !important; }
.ocr-status { margin-top:10px !important; padding:10px 12px !important; border:1px solid #1A1A1A !important; background:#F4EFE4 !important; color:#1A1A1A !important; font-family:'Courier New',monospace !important; font-size:12px !important; line-height:1.45 !important; }
.ocr-status--success { border-color:#1A1A1A !important; background:#E8F6EF !important; color:#0B6E4F !important; }
.ocr-status--warning { border-color:#1A1A1A !important; background:#FFF1D8 !important; color:#8A5A00 !important; }
.ocr-status--info { border-color:#1A1A1A !important; background:#EEF2F0 !important; color:#555 !important; }
.ocr-results { display:flex !important; flex-wrap:wrap !important; gap:8px !important; margin-top:10px !important; }
.ocr-result { display:inline-flex !important; align-items:center !important; padding:4px 10px !important; border:1px solid #1A1A1A !important; background:#FFFFFF !important; color:#1A1A1A !important; font-family:'Courier New',monospace !important; font-size:11px !important; }
.ocr-result--amount { border-color:#1A1A1A !important; color:#1A1A1A !important; }
.ocr-result--upi { border-color:#1A1A1A !important; color:#1A1A1A !important; }
[data-testid="stAlert"] { border-radius:0 !important; }
[data-testid="stPlotlyChart"] { animation:chartRise .55s cubic-bezier(0.16,1,0.3,1) both; }
[data-testid="stPlotlyChart"]:nth-child(1){animation-delay:.00s}
[data-testid="stPlotlyChart"]:nth-child(2){animation-delay:.12s}
[data-testid="stPlotlyChart"]:nth-child(3){animation-delay:.24s}
[data-testid="stPlotlyChart"]:nth-child(4){animation-delay:.36s}
[data-testid="stPlotlyChart"]:nth-child(5){animation-delay:.48s}
[data-testid="stPlotlyChart"]:nth-child(6){animation-delay:.60s}

.hero-banner { background:#1A1A1A !important; border-radius:0 !important; padding:24px 28px !important; margin-bottom:0 !important; animation:fadeInUp .5s ease both; }
.hero-banner h1 { color:#F4EFE4 !important; font-family:'Times New Roman',serif !important; font-style:italic !important; font-size:2.2rem !important; letter-spacing:-.02em !important; margin:0 0 6px 0 !important; }
.hero-banner p { color:#AAA !important; font-family:'Courier New',monospace !important; font-size:.8rem !important; letter-spacing:.14em; margin:0 !important; }
.hero-banner::before { display:none !important; }
.hero-badge { background:rgba(255,255,255,.1) !important; border:1px solid rgba(255,255,255,.2) !important; color:#CCC !important; border-radius:0 !important; font-family:'Courier New',monospace !important; font-size:.7rem !important; letter-spacing:.14em; padding:2px 8px !important; }
.kpi-card { border-radius:0 !important; border:1.5px solid #1A1A1A !important; border-left-width:1.5px !important; background:#F4EFE4 !important; text-align:center; animation:slideIn .4s ease both; }
.kpi-blue,.kpi-red,.kpi-green,.kpi-amber { background:#F4EFE4 !important; border-color:#1A1A1A !important; }
.kpi-value { color:#1A1A1A !important; font-family:'Courier New',monospace !important; font-size:1.5rem !important; }
.kpi-label { color:#666 !important; font-family:'Courier New',monospace !important; letter-spacing:.14em; font-size:10px !important; }
.kpi-icon { display:none; }
.section-header { color:#1A1A1A !important; letter-spacing:.22em !important; font-family:'Courier New',monospace !important; font-size:10px !important; border-bottom:1px solid #1A1A1A; padding-bottom:3px; animation:chartSlideLeft .4s ease both; }
.verdict-fraud { background:#1A1A1A !important; border-radius:0 !important; box-shadow:none !important; border:3px solid #1A1A1A !important; padding:24px !important; animation:chartFadeScale .45s cubic-bezier(0.34,1.56,0.64,1) both; }
.verdict-fraud h2,.verdict-fraud p,.verdict-fraud strong { color:#F4EFE4 !important; }
.verdict-normal { background:#F4EFE4 !important; border-radius:0 !important; box-shadow:none !important; border:3px solid #1A1A1A !important; padding:24px !important; animation:chartFadeScale .45s cubic-bezier(0.34,1.56,0.64,1) both; }
.verdict-normal h2,.verdict-normal p,.verdict-normal strong { color:#1A1A1A !important; }
.fraud-tag { background:transparent !important; border:1.5px solid #1A1A1A !important; color:#1A1A1A !important; border-radius:0 !important; font-family:'Courier New',monospace !important; font-size:11px !important; padding:2px 8px !important; margin:2px !important; display:inline-block !important; }
.stat-card { background:#F4EFE4 !important; border:1.5px solid #1A1A1A !important; border-radius:0 !important; animation:slideIn .5s ease both; }
.stat-card h2 { font-family:'Courier New',monospace !important; background:none !important; -webkit-text-fill-color:#1A1A1A !important; font-size:2rem !important; }
.stat-card p { color:#666 !important; font-size:12px !important; }
.pipeline-step { background:#F4EFE4 !important; border:1.5px solid #1A1A1A !important; border-radius:0 !important; transition:box-shadow .2s !important; }
.pipeline-step:hover { border-color:#1A1A1A !important; box-shadow:3px 3px 0 #1A1A1A !important; }
""",

'swiss': """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;900&family=Barlow:wght@400;500;600&display=swap');
@keyframes fadeInUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes slideRight { from{opacity:0;transform:translateX(-10px)} to{opacity:1;transform:translateX(0)} }
@keyframes chartRise { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }
@keyframes chartFadeScale { from{opacity:0;transform:scale(0.96)} to{opacity:1;transform:scale(1)} }
@keyframes chartSlideLeft { from{opacity:0;transform:translateX(-20px)} to{opacity:1;transform:translateX(0)} }

html, body, [class*="css"] { background:#FFFFFF !important; color:#111 !important; font-family:'Helvetica Neue',Helvetica,Arial,sans-serif !important; font-size:14px !important; }
[data-testid="stAppViewContainer"],[data-testid="stMain"] { background:#FFFFFF !important; }
.main .block-container { background:#FFFFFF !important; max-width:1400px !important; padding:1.5rem 2rem !important; }
[data-testid="stSidebar"],[data-testid="stSidebar"] > div { background:#FFFFFF !important; border-right:4px solid #E8320A !important; }
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,[data-testid="stSidebar"] label { color:#444 !important; font-weight:600; letter-spacing:.04em; font-size:12px !important; }
[data-testid="stMarkdown"] p,[data-testid="stMarkdownContainer"] p,[data-testid="stText"] { color:#111 !important; font-size:14px !important; }
[data-testid="stMarkdown"] h1,[data-testid="stMarkdown"] h2,[data-testid="stMarkdown"] h3,[data-testid="stMarkdown"] h4 { color:#111 !important; font-weight:700 !important; }
[data-testid="metric-container"] { background:#FFFFFF !important; border:1.5px solid #111 !important; border-radius:0 !important; box-shadow:none !important; padding:16px !important; animation:slideRight .4s ease both; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'Courier New',monospace !important; color:#111 !important; font-weight:700; font-size:1.7rem !important; }
[data-testid="metric-container"] [data-testid="stMetricLabel"] p { color:#888 !important; letter-spacing:.18em; text-transform:uppercase; font-size:9px !important; font-weight:700; }
.stTabs [data-baseweb="tab-list"] { background:#FFF !important; border-bottom:3px solid #111; }
.stTabs [data-baseweb="tab"] { background:transparent !important; border:1.5px solid #111 !important; border-radius:0 !important; color:#111 !important; margin-right:-1px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; font-size:11px !important; padding:8px 18px !important; transition:all .15s !important; }
.stTabs [aria-selected="true"] { background:#111 !important; color:#FFF !important; box-shadow:none !important; }
.stButton > button { background:transparent !important; border:1.5px solid #111 !important; color:#111 !important; border-radius:0 !important; font-weight:700; letter-spacing:.1em; text-transform:uppercase; font-size:11px !important; padding:6px 14px !important; transition:all .15s !important; }
.stButton > button:hover { background:#111 !important; color:#FFF !important; }
.stButton > button[kind="primary"] { background:#E8320A !important; border:none !important; color:#FFF !important; border-radius:0 !important; box-shadow:none !important; }
.stButton > button[kind="primary"]:hover { background:#111 !important; }
[data-testid="stSidebar"] .stButton > button { background:transparent !important; border:1.5px solid #111 !important; color:#111 !important; border-radius:0 !important; font-weight:700; letter-spacing:.1em; text-transform:uppercase; font-size:11px !important; padding:6px 14px !important; }
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button div { color:#111 !important; }
[data-testid="stSidebar"] .stButton > button:hover { background:#111 !important; color:#FFF !important; }
[data-testid="stSidebar"] .stButton > button:hover span,
[data-testid="stSidebar"] .stButton > button:hover p { color:#FFF !important; }
[data-testid="stDataFrame"] { border-radius:0 !important; border:1.5px solid #111 !important; animation:chartFadeScale .4s ease both; animation-delay:.1s; }
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th { color:#111 !important; font-size:12px !important; }
.stSlider [data-testid="stMarkdownContainer"] p { color:#444 !important; font-weight:600; letter-spacing:.06em; text-transform:uppercase; font-size:10px !important; }
[data-testid="stSelectbox"] label p { color:#444 !important; font-weight:700; letter-spacing:.06em; text-transform:uppercase; font-size:11px !important; }
[data-testid="stSelectbox"] > div > div { background:#FFF !important; border:1.5px solid #111 !important; color:#111 !important; border-radius:0 !important; }
.stExpander { border:1.5px solid #111 !important; border-radius:0 !important; }
.stExpander summary p { color:#111 !important; font-weight:700; }
[data-testid="stHorizontalRule"] { border-color:#E0E0E0 !important; }
[data-testid="stAlert"] p { color:#444 !important; }
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input { background:#FFF !important; border:1.5px solid #111 !important; color:#111 !important; border-radius:0 !important; font-weight:500; }
[data-testid="stFileUploader"] { border:1.5px dashed #111 !important; background:#FFF !important; border-radius:0 !important; }
[data-testid="stFileUploader"] p { color:#444 !important; }
[data-testid="stCaption"] p { color:#888 !important; font-size:11px !important; letter-spacing:.04em; }
.stRadio label p { color:#444 !important; font-weight:600; }
.stCheckbox label p { color:#444 !important; font-weight:600; }
[data-testid="stFileUploader"] section { padding:10px 12px !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] { min-height:84px !important; }
[data-testid="stFileUploader"] [data-testid="stFileUploadDropzone"] > div { padding:4px 0 !important; }
[data-testid="stFileUploader"] button { border-radius:0 !important; }
.ocr-panel { background:#FFF !important; border:1.5px solid #111 !important; padding:16px 18px !important; margin:8px 0 12px 0 !important; }
.ocr-panel__title { color:#111 !important; font-family:'Helvetica Neue',sans-serif !important; font-size:12px !important; letter-spacing:.22em !important; text-transform:uppercase !important; margin:0 0 6px 0 !important; font-weight:700 !important; }
.ocr-panel__subtitle { color:#444 !important; font-family:'Helvetica Neue',sans-serif !important; font-size:12px !important; margin:0 !important; line-height:1.5 !important; }
.ocr-panel__meta { display:flex !important; flex-wrap:wrap !important; gap:6px !important; margin-top:10px !important; }
.ocr-chip { display:inline-flex !important; align-items:center !important; gap:6px !important; padding:3px 8px !important; border:1.5px solid #111 !important; background:#FFF !important; color:#111 !important; font-family:'Helvetica Neue',sans-serif !important; font-size:10px !important; letter-spacing:.1em !important; text-transform:uppercase !important; }
.ocr-chip--accent { border-color:#E8320A !important; color:#E8320A !important; background:rgba(232,50,10,.06) !important; }
.ocr-chip--muted { color:#666 !important; }
.ocr-status { margin-top:10px !important; padding:10px 12px !important; border:1.5px solid #111 !important; background:#FFF !important; color:#111 !important; font-family:'Helvetica Neue',sans-serif !important; font-size:12px !important; line-height:1.45 !important; }
.ocr-status--success { border-color:#111 !important; background:#F3F8F4 !important; color:#111 !important; }
.ocr-status--warning { border-color:#111 !important; background:#FFF6EA !important; color:#111 !important; }
.ocr-status--info { border-color:#111 !important; background:#FAFAFA !important; color:#555 !important; }
.ocr-results { display:flex !important; flex-wrap:wrap !important; gap:8px !important; margin-top:10px !important; }
.ocr-result { display:inline-flex !important; align-items:center !important; padding:4px 10px !important; border:1.5px solid #111 !important; background:#FFF !important; color:#111 !important; font-family:'Helvetica Neue',sans-serif !important; font-size:11px !important; }
.ocr-result--amount { border-color:#111 !important; color:#111 !important; }
.ocr-result--upi { border-color:#E8320A !important; color:#E8320A !important; }
[data-testid="stAlert"] { border-radius:0 !important; }
[data-testid="stPlotlyChart"] { animation:chartRise .55s cubic-bezier(0.16,1,0.3,1) both; }
[data-testid="stPlotlyChart"]:nth-child(1){animation-delay:.00s}
[data-testid="stPlotlyChart"]:nth-child(2){animation-delay:.12s}
[data-testid="stPlotlyChart"]:nth-child(3){animation-delay:.24s}
[data-testid="stPlotlyChart"]:nth-child(4){animation-delay:.36s}
[data-testid="stPlotlyChart"]:nth-child(5){animation-delay:.48s}
[data-testid="stPlotlyChart"]:nth-child(6){animation-delay:.60s}

.hero-banner { background:#111 !important; border-radius:0 !important; padding:24px 28px !important; margin-bottom:0 !important; border-bottom:4px solid #E8320A !important; animation:fadeInUp .5s ease both; }
.hero-banner h1 { color:#FFF !important; font-family:'Helvetica Neue',sans-serif !important; font-size:2rem !important; font-weight:700 !important; letter-spacing:.28em !important; text-transform:uppercase !important; margin:0 0 6px 0 !important; }
.hero-banner p { color:#888 !important; font-size:.85rem !important; letter-spacing:.06em; margin:0 !important; }
.hero-banner::before { display:none !important; }
.hero-badge { background:transparent !important; border:1.5px solid #E8320A !important; color:#E8320A !important; border-radius:0 !important; font-weight:700; letter-spacing:.14em; font-size:.7rem !important; padding:2px 8px !important; }
.kpi-card { border-radius:0 !important; border:1.5px solid #E0E0E0 !important; border-left-width:4px !important; background:#FFF !important; animation:slideRight .4s ease both; }
.kpi-blue { border-left-color:#111 !important; }
.kpi-red   { border-left-color:#E8320A !important; }
.kpi-green { border-left-color:#111 !important; }
.kpi-amber { border-left-color:#E8320A !important; }
.kpi-value { color:#111 !important; font-family:'Courier New',monospace !important; font-weight:700; font-size:1.5rem !important; }
.kpi-label { color:#888 !important; letter-spacing:.18em; text-transform:uppercase; font-weight:700; font-size:10px !important; }
.section-header { color:#E8320A !important; letter-spacing:.28em !important; font-weight:700 !important; font-size:10px !important; animation:chartSlideLeft .4s ease both; }
.verdict-fraud { background:linear-gradient(135deg,#E8320A,#111) !important; border-radius:0 !important; box-shadow:none !important; border:none !important; padding:24px !important; animation:chartFadeScale .45s cubic-bezier(0.34,1.56,0.64,1) both; }
.verdict-fraud h2,.verdict-fraud p,.verdict-fraud strong { color:#FFF !important; }
.verdict-normal { background:#FFF !important; border-radius:0 !important; box-shadow:none !important; border:3px solid #111 !important; padding:24px !important; animation:chartFadeScale .45s cubic-bezier(0.34,1.56,0.64,1) both; }
.verdict-normal h2,.verdict-normal p,.verdict-normal strong { color:#111 !important; }
.fraud-tag { background:rgba(232,50,10,.08) !important; border:1.5px solid #E8320A !important; color:#E8320A !important; border-radius:0 !important; font-weight:700; letter-spacing:.08em; font-size:11px !important; padding:2px 8px !important; margin:2px !important; display:inline-block !important; }
.stat-card { background:#FFF !important; border:1.5px solid #111 !important; border-radius:0 !important; animation:slideRight .5s ease both; }
.stat-card::after { background:#E8320A !important; }
.stat-card h2 { font-family:'Courier New',monospace !important; background:none !important; -webkit-text-fill-color:#111 !important; font-weight:700; font-size:2rem !important; }
.stat-card p { color:#888 !important; font-size:12px !important; }
.pipeline-step { background:#FFF !important; border:1.5px solid #111 !important; border-radius:0 !important; transition:all .2s !important; }
.pipeline-step:hover { border-color:#E8320A !important; box-shadow:3px 3px 0 #E8320A !important; }
"""
}

st.markdown(f"""
<style>
{_THEME_CSS[_theme]}

/* ══ HIDE STREAMLIT TOOLBAR / HEADER / FOOTER ══ */
#MainMenu {{ visibility: hidden !important; display: none !important; }}
header[data-testid="stHeader"] {{
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  height: 0 !important;
  min-height: 0 !important;
  background: transparent !important;
  pointer-events: none !important;
  z-index: 2147483646 !important;
}}
header[data-testid="stHeader"] button,
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {{
  pointer-events: auto !important;
}}
footer {{ visibility: hidden !important; height: 0 !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stStatusWidget"],
[data-testid="manage-app-button"],
header[data-testid="stHeader"] [data-testid="stToolbar"],
header[data-testid="stHeader"] [data-testid="stDecoration"] {{
  display: none !important;
}}
[data-testid="stSidebar"] {{
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  min-width: 18rem !important;
  width: 18rem !important;
  z-index: 999998 !important;
}}
[data-testid="stSidebar"] > div:first-child {{
  width: 18rem !important;
  min-width: 18rem !important;
}}
[data-testid="collapsedControl"] {{ display: flex !important; visibility: visible !important; opacity: 1 !important; position: fixed !important; top: 1rem !important; left: 1rem !important; z-index: 999999 !important; background: #FF8C00 !important; border-radius: 4px !important; }}
[data-testid="collapsedControl"] button {{ color: #000 !important; }}
[data-testid="collapsedControl"] svg {{ fill: #000 !important; }}
[data-testid="stSidebarCollapsedControl"],
button[title="Open sidebar"],
button[aria-label="Open sidebar"] {{
  display: inline-flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  position: fixed !important;
  top: .85rem !important;
  left: .85rem !important;
  z-index: 2147483647 !important;
  background: rgba(255,255,255,.96) !important;
  border: 1px solid rgba(17,17,17,.22) !important;
  border-radius: 8px !important;
  color: #111 !important;
}}

/* ── Shared ── */
.stSpinner > div {{ border-color: #FF8C00 transparent transparent transparent !important; }}
.score-badge {{ font-family: 'Courier New', monospace; font-size: 0.8rem; font-weight: 500; padding: 2px 8px; }}
.score-high {{ background: rgba(224,62,62,0.1); color: #E03E3E !important; }}
.score-mid  {{ background: rgba(232,144,10,0.1); color: #E8900A !important; }}
.score-low  {{ background: rgba(29,122,79,0.1);  color: #1D7A4F !important; }}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span {{
  color: inherit !important;
}}
.stButton > button p,
.stButton > button span,
.stDownloadButton > button p,
.stDownloadButton > button span {{
  color: inherit !important;
}}
</style>
""", unsafe_allow_html=True)
import streamlit.components.v1 as _head_css
_head_css.html("""
<script>
const style = document.createElement('style');
style.textContent = `
  #payguard-sidebar-toggle {
    align-items: center !important;
    background: #FF8C00 !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 8px !important;
    box-shadow: 0 8px 24px rgba(0,0,0,0.28) !important;
    color: #07090D !important;
    cursor: pointer !important;
    display: inline-flex !important;
    font: 700 16px/1 "Courier New", monospace !important;
    height: 34px !important;
    justify-content: center !important;
    left: 12px;
    opacity: 0.96 !important;
    position: fixed !important;
    top: 12px !important;
    transition: left .18s ease, filter .18s ease !important;
    width: 34px !important;
    z-index: 2147483647 !important;
  }
  #payguard-sidebar-toggle:hover {
    filter: brightness(1.08) !important;
  }
  [data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 1rem !important;
    left: 1rem !important;
    z-index: 2147483647 !important;
    background: rgba(255,140,0,0.15) !important;
    border-radius: 4px !important;
  }
  [data-testid="collapsedControl"] svg {
    fill: #FF8C00 !important;
  }
`;
document.head.appendChild(style);

function installPayGuardSidebarToggle() {
  const doc = window.parent && window.parent.document ? window.parent.document : document;
  if (!doc || doc.getElementById('payguard-sidebar-toggle')) return;

  const btn = doc.createElement('button');
  btn.id = 'payguard-sidebar-toggle';
  btn.type = 'button';
  btn.setAttribute('aria-label', 'Toggle sidebar');
  btn.setAttribute('title', 'Toggle sidebar');
  btn.textContent = '☰';
  doc.body.appendChild(btn);

  const styleNode = doc.createElement('style');
  styleNode.textContent = style.textContent;
  doc.head.appendChild(styleNode);

  const findNativeToggle = () => (
    doc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
    doc.querySelector('[data-testid="collapsedControl"] button') ||
    doc.querySelector('[data-testid="stSidebarCollapsedControl"] button') ||
    doc.querySelector('button[aria-label="Open sidebar"]') ||
    doc.querySelector('button[title="Open sidebar"]') ||
    doc.querySelector('button[aria-label="Close sidebar"]') ||
    doc.querySelector('button[title="Close sidebar"]')
  );

  btn.addEventListener('click', () => {
    const nativeToggle = findNativeToggle();
    if (nativeToggle) nativeToggle.click();
  });

  const syncPosition = () => {
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    const rect = sidebar ? sidebar.getBoundingClientRect() : null;
    const isOpen = rect && rect.width > 80 && rect.left > -80;
    btn.style.left = isOpen ? `${Math.round(rect.right + 10)}px` : '12px';
  };

  syncPosition();
  setInterval(syncPosition, 350);
}

installPayGuardSidebarToggle();
setInterval(installPayGuardSidebarToggle, 1000);
</script>
""", height=0)
# ── Constants ──
UPI_APPS      = ['GPay','PhonePe','Paytm','BHIM','AmazonPay','WhatsApp']
BANKS         = ['SBI','HDFC','ICICI','Axis','Kotak','PNB','BOB','Canara','IndusInd','Yes Bank','IDFC First','UCO Bank']
STATES        = ['Maharashtra','Karnataka','Delhi','Tamil Nadu','Telangana','Gujarat','Rajasthan','UP','West Bengal','Kerala','MP','Bihar','Odisha','Punjab','Haryana']
MERCHANT_CATS = ['p2p','food','utilities','groceries','fuel','recharge','ecommerce','rent','education']

# ── Sidebar ──
with st.sidebar:
    st.markdown("### 🔷 PayGuard")
    st.markdown("<p style='color:#6B7280;font-size:0.8rem;margin-top:-8px'>Real-time UPI fraud monitoring</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio('Navigation', ['🏠 Overview','📡 Live Score','📤 Scan','📊 Analytics','ℹ️ About'], label_visibility='collapsed')
    st.divider()
    st.markdown("<p class='section-header'>UI Theme</p>", unsafe_allow_html=True)
    st.markdown("""
<style>
[data-testid="stSidebar"] div[data-testid="column"] .stButton > button {
    align-items: center !important;
    aspect-ratio: 1 / 1 !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    display: inline-flex !important;
    font-size: 0 !important;
    height: 26px !important;
    justify-content: center !important;
    line-height: 1 !important;
    padding: 0 !important;
    transition: border-color .16s ease, background .16s ease, transform .16s ease !important;
}
[data-testid="stSidebar"] div[data-testid="column"] .stButton > button[kind="primary"] {
    box-shadow: 0 0 0 1px rgba(255, 255, 255, .38), 0 0 0 3px rgba(255, 140, 0, .18) !important;
}
[data-testid="stSidebar"] div[data-testid="column"] .stButton > button p,
[data-testid="stSidebar"] div[data-testid="column"] .stButton > button span {
    font-size: 11px !important;
    line-height: 1 !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] div[data-testid="column"]:nth-of-type(1) .stButton > button {
    background: #FF8C00 !important;
    border: 1px solid #FFB45A !important;
    color: #07090D !important;
}
[data-testid="stSidebar"] div[data-testid="column"]:nth-of-type(2) .stButton > button {
    background: #1A1A1A !important;
    border: 1px solid #555 !important;
    color: #F4EFE4 !important;
}
[data-testid="stSidebar"] div[data-testid="column"]:nth-of-type(3) .stButton > button {
    background: #FFFFFF !important;
    border: 1px solid #E8320A !important;
    color: #E8320A !important;
}
[data-testid="stSidebar"] div[data-testid="column"] .stButton > button:hover {
    filter: brightness(1.06) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] button[kind="secondary"] {
    background: rgba(255,255,255,.08) !important;
    border: 1px solid rgba(138,171,184,.28) !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    color: #8AABB8 !important;
    min-height: 24px !important;
    height: 26px !important;
    padding: 0 !important;
}
[data-testid="stSidebar"] button[kind="primary"] {
    border-color: rgba(255,140,0,.48) !important;
    box-shadow: 0 0 0 2px rgba(255,140,0,.12) !important;
    color: #FF8C00 !important;
}
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] button[kind="secondary"] p {
    color: inherit !important;
    font-size: 11px !important;
    line-height: 1 !important;
}
</style>
""", unsafe_allow_html=True)
    _t_cols = st.columns(3)
    _theme_ids = list(THEMES.keys())
    _theme_icons = {'bloomberg': '■', 'broadsheet': '●', 'swiss': '◆'}
    for _ti, (_tcol, _tid) in enumerate(zip(_t_cols, _theme_ids)):
        _button_type = 'primary' if _tid == _theme else 'secondary'
        if _tcol.button(_theme_icons[_tid], key=f'theme_{_tid}', help=THEMES[_tid]['label'], type=_button_type, use_container_width=True):
            st.session_state['ui_theme'] = _tid
            st.rerun()
    st.markdown(f"<p style='font-size:9px;color:#888;letter-spacing:.08em;text-transform:uppercase;margin-top:4px'>Active: {THEMES[_theme]['label']}</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p class='section-header'>Settings</p>", unsafe_allow_html=True)
    model_choice = st.selectbox('Model', ['Isolation Forest','Autoencoder','LOF','Ensemble'])
    _model_defaults = {
        'Isolation Forest': IF_THRESHOLD,
        'Autoencoder':      AE_THRESHOLD,
        'LOF':              LOF_THRESHOLD,
        'Ensemble':         THRESHOLD,
    }
    threshold = st.slider('Anomaly Threshold', 0.0, 1.0,
                          _model_defaults.get(model_choice, THRESHOLD), 0.01,
                          help='Normalised anomaly score [0–1]. Default auto-set per model.')
    st.divider()
    st.markdown("<p style='color:#6B7280;font-size:0.75rem'>PayGuard · Unsupervised ML · 293k transactions</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280;font-size:0.65rem'>Presentation build · Local analytics workspace</p>", unsafe_allow_html=True)


# ── Cached loaders ──
@st.cache_resource
def load_if_model():
    import pickle
    with open(IF_MODEL_PATH, 'rb') as f: model = pickle.load(f)
    with open(IF_SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def load_ae_model(input_dim):
    import pickle, torch
    from src.models import Autoencoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae = Autoencoder(input_dim, AE_HIDDEN).to(device)
    ae.load_state_dict(torch.load(AE_MODEL_PATH, map_location=device, weights_only=True))
    ae.eval()
    with open(AE_SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    return ae, scaler, device

@st.cache_resource
def load_lof_model():
    import pickle
    with open(LOF_MODEL_PATH, 'rb') as f: lof_model = pickle.load(f)
    with open(LOF_SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    return lof_model, scaler

@st.cache_data
def load_sender_stats():
    if os.path.exists(SENDER_STATS_PATH):
        return pd.read_csv(SENDER_STATS_PATH).set_index('sender_upi')
    return None

@st.cache_data
def load_receiver_stats():
    p = os.path.join(OUTPUTS_DIR, 'receiver_stats.csv')
    if os.path.exists(p):
        return pd.read_csv(p).set_index('receiver_upi')
    return None

@st.cache_resource
def load_velocity_store():
    try:
        from src.velocity_store import VelocityStore
        return VelocityStore()
    except Exception:
        return None

@st.cache_resource
def load_feature_scaler():
    import pickle
    p = os.path.join(OUTPUTS_DIR, 'feature_scaler.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f: return pickle.load(f)
    return None


# ── Helpers ──
def _inverse_scale_amount(scaled_values):
    feat_scaler = load_feature_scaler()
    if feat_scaler is not None:
        try:
            return scaled_values * feat_scaler.scale_[0] + feat_scaler.mean_[0]
        except Exception:
            pass
    return scaled_values

def _score_to_risk(score, threshold):
    if score >= threshold: return '🔴 High Risk'
    elif score >= threshold * 0.6: return '🟡 Medium'
    return '🟢 Low Risk'

def encode_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['amount']     = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    out['amount_log'] = np.log1p(out['amount'])
    sender_stats = load_sender_stats()
    historical_global_mean = sender_stats['amount_mean'].mean() if sender_stats is not None and len(sender_stats) > 0 else 1.0
    global_mean = out['amount'].mean() if len(out) > 1 else historical_global_mean
    if sender_stats is not None and 'sender_upi' in df.columns:
        known_avg = df['sender_upi'].map(sender_stats['amount_mean'])
        sender_avg = known_avg.fillna(global_mean)
    elif 'sender_upi' in df.columns:
        sender_avg = df.groupby('sender_upi')['amount'].transform('mean')
    else:
        sender_avg = pd.Series(global_mean, index=df.index)
    out['amount_vs_sender_avg'] = out['amount'] / (pd.to_numeric(sender_avg, errors='coerce').fillna(global_mean) + 1)
    out['is_round_amount'] = (out['amount'] % 1000 == 0).astype(int)
    try:
        ts = pd.to_datetime(df['timestamp'])
        out['txn_hour'] = ts.dt.hour
        out['txn_day']  = ts.dt.dayofweek
    except Exception:
        out['txn_hour'] = 12
        out['txn_day']  = 0
    out['is_weekend'] = (out['txn_day'] >= 5).astype(int)
    out['is_night']   = ((out['txn_hour'] >= 23) | (out['txn_hour'] <= 4)).astype(int)
    if 'sender_upi' in df.columns:
        try:
            _vs = load_velocity_store()
            if _vs is None: raise Exception('VelocityStore unavailable')
            if 'timestamp' in df.columns:
                v1h_s, v24h_s, urecv_s, entr_s = _vs.compute_from_df(df)
                out['velocity_1h']         = v1h_s.clip(lower=0).fillna(1).astype(int)
                out['velocity_24h']        = v24h_s.clip(lower=0).fillna(1).astype(int)
                out['unique_receivers_1h'] = urecv_s.clip(lower=0).fillna(0).astype(int)
                out['amount_entropy_1h']   = entr_s.clip(lower=0).fillna(0.0)
            else:
                vel_dicts = [_vs.get(s, __import__('time').time()) for s in df['sender_upi']]
                out['velocity_1h']         = [d['velocity_1h']         for d in vel_dicts]
                out['velocity_24h']        = [d['velocity_24h']        for d in vel_dicts]
                out['unique_receivers_1h'] = [d['unique_receivers_1h'] for d in vel_dicts]
                out['amount_entropy_1h']   = [d['amount_entropy_1h']   for d in vel_dicts]
        except Exception:
            vc = df['sender_upi'].value_counts()
            out['velocity_1h'] = df['sender_upi'].map(vc).fillna(1).astype(int)
            out['velocity_24h'] = out['velocity_1h']
            out['unique_receivers_1h'] = 0
            out['amount_entropy_1h']   = 0.0
    else:
        out['velocity_1h'] = out['velocity_24h'] = 1
        out['unique_receivers_1h'] = 0
        out['amount_entropy_1h']   = 0.0
    device_registry_path = os.path.join(OUTPUTS_DIR, 'device_registry.pkl')
    if 'device_id' in df.columns and os.path.exists(device_registry_path):
        import pickle as _pkl
        with open(device_registry_path, 'rb') as _f: _dev_reg = _pkl.load(_f)
        out['is_new_device'] = df.apply(lambda r: int(r['device_id'] not in _dev_reg.get(r['sender_upi'], set())), axis=1)
    else:
        out['is_new_device'] = 0
    out['cross_bank']  = (df['sender_bank']  != df['receiver_bank']).astype(int)  if ('sender_bank'  in df.columns and 'receiver_bank'  in df.columns) else 0
    out['cross_state'] = (df['sender_state'] != df['receiver_state']).astype(int) if ('sender_state' in df.columns and 'receiver_state' in df.columns) else 0
    receiver_stats = load_receiver_stats()
    if receiver_stats is not None and 'receiver_upi' in df.columns:
        recv_count = df['receiver_upi'].map(receiver_stats['receiver_txn_count_24h_mean'])   if 'receiver_txn_count_24h_mean'   in receiver_stats.columns else pd.Series(0,   index=df.index)
        recv_amt   = df['receiver_upi'].map(receiver_stats['receiver_amount_sum_24h_mean'])  if 'receiver_amount_sum_24h_mean'  in receiver_stats.columns else pd.Series(0.0, index=df.index)
        out['receiver_txn_count_24h']  = recv_count.fillna(0).clip(lower=0)
        out['receiver_amount_sum_24h'] = recv_amt.fillna(0).clip(lower=0)
    else:
        out['receiver_txn_count_24h']  = 0
        out['receiver_amount_sum_24h'] = 0.0
    out['upi_app_enc']        = pd.Categorical(df['upi_app']           if 'upi_app'           in df.columns else pd.Series('GPay',  index=df.index), categories=UPI_APPS).codes
    out['merchant_cat_enc']   = pd.Categorical(df['merchant_category'] if 'merchant_category' in df.columns else pd.Series('p2p',   index=df.index), categories=MERCHANT_CATS).codes
    out['sender_bank_enc']    = pd.Categorical(df['sender_bank']       if 'sender_bank'       in df.columns else pd.Series('SBI',   index=df.index), categories=BANKS).codes
    out['receiver_bank_enc']  = pd.Categorical(df['receiver_bank']     if 'receiver_bank'     in df.columns else pd.Series('HDFC',  index=df.index), categories=BANKS).codes
    out['sender_state_enc']   = pd.Categorical(df['sender_state']      if 'sender_state'      in df.columns else pd.Series('Delhi', index=df.index), categories=STATES).codes
    out['receiver_state_enc'] = pd.Categorical(df['receiver_state']    if 'receiver_state'    in df.columns else pd.Series('Delhi', index=df.index), categories=STATES).codes
    out = out.fillna(0)
    _SCALE_COLS = ['amount','amount_log','amount_vs_sender_avg','velocity_1h','velocity_24h','txn_hour','txn_day','unique_receivers_1h','amount_entropy_1h','receiver_txn_count_24h','receiver_amount_sum_24h']
    feat_scaler = load_feature_scaler()
    if feat_scaler is not None:
        out[_SCALE_COLS] = feat_scaler.transform(out[_SCALE_COLS])
    return out

def is_raw_upi(df):
    return 'sender_upi' in df.columns or ('amount' in df.columns and 'upi_app' in df.columns)

def get_fraud_tags(row):
    tags = []
    hour = row.get('txn_hour', -1)
    if row.get('is_night', 0) or (0 <= hour <= 4) or hour >= 23: tags.append('🌙 Late Night')
    if row.get('is_new_device', 0):                               tags.append('📱 New Device')
    if row.get('amount_vs_sender_avg', 1) > 3:                    tags.append('💰 High Amount')
    if row.get('velocity_1h', 0) >= 4:                            tags.append('⚡ High Velocity')
    if row.get('is_round_amount', 0):                             tags.append('🔄 Round Amount')
    if row.get('cross_state', 0):                                 tags.append('🗺️ Cross State')
    if row.get('cross_bank', 0):                                  tags.append('🏦 Cross Bank')
    if not tags:                                                   tags.append('⚠️ Statistical Outlier')
    return tags

def format_inr(x):
    if x >= 1e7:  return f"₹{x/1e7:.1f}Cr"
    elif x >= 1e5: return f"₹{x/1e5:.1f}L"
    elif x >= 1e3: return f"₹{x/1e3:.1f}K"
    return f"₹{x:.0f}"

SCORE_MAP = {'Isolation Forest':'if_score','Autoencoder':'ae_score','LOF':'lof_score','Ensemble':'ensemble_score'}
COLORS    = {'Isolation Forest':'#4C9EFF','Autoencoder':'#4CC9F0','LOF':'#F7B731','Ensemble':'#00C896'}

@st.cache_data(show_spinner=False)
def dataset_split_counts(test_count: int):
    total_count = test_count
    if os.path.exists(DATA_PATH):
        total_count = len(pd.read_csv(DATA_PATH, usecols=['amount']))
    val_count = len(pd.read_csv(X_VAL_PATH, usecols=['amount'])) if os.path.exists(X_VAL_PATH) else 0
    train_count = max(total_count - val_count - test_count, 0)
    return total_count, train_count, val_count, test_count

@st.cache_data(show_spinner=False)
def cached_inference(feature_df_json: str, model_name: str, threshold: float):
    from src.models import run_inference
    feature_df = pd.read_json(feature_df_json, dtype=float)
    return run_inference(feature_df, model_name=model_name, threshold=threshold)

def light_chart(fig, height=400):
    if _theme == 'bloomberg':
        bg, plot_bg, fg, grid, leg = '#07090D', '#07090D', '#C0C8D0', '#1A2430', 'rgba(7,9,13,0.9)'
        font_family = 'Courier New'
    elif _theme == 'broadsheet':
        bg, plot_bg, fg, grid, leg = '#F4EFE4', '#F4EFE4', '#1A1A1A', '#C8C0A8', 'rgba(244,239,228,0.9)'
        font_family = 'Times New Roman'
    else:
        bg, plot_bg, fg, grid, leg = '#FFFFFF', '#FFFFFF', '#111111', '#E0E0E0', 'rgba(255,255,255,0.9)'
        font_family = 'Helvetica Neue'
    fig.update_layout(
        height=height, paper_bgcolor=bg, plot_bgcolor=plot_bg,
        font=dict(color=fg, family=font_family),
        xaxis=dict(showgrid=True, gridcolor=grid, color=fg, zerolinecolor=grid),
        yaxis=dict(showgrid=True, gridcolor=grid, color=fg, zerolinecolor=grid),
        legend=dict(bgcolor=leg, font=dict(color=fg)),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == '🏠 Overview':
    _has_data = os.path.exists(ANOMALY_SCORES_PATH) and os.path.exists(Y_TEST_PATH)
    if _has_data:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        scores = pd.read_csv(ANOMALY_SCORES_PATH)
        y_true = pd.read_csv(Y_TEST_PATH).values.ravel()
        X_test = pd.read_csv(X_TEST_PATH)
        _n = min(len(scores), len(y_true), len(X_test))
        scores = scores.iloc[:_n].reset_index(drop=True)
        y_true = y_true[:_n]
        X_test = X_test.iloc[:_n].reset_index(drop=True)
        sc     = SCORE_MAP[model_choice]
        y_pred = (scores[sc] >= threshold).astype(int)
        n_flagged = int(y_pred.sum())
        n_total   = len(y_pred)
        dataset_total, train_total, val_total, test_total = dataset_split_counts(n_total)
        if 'amount' in X_test.columns:
            _real_amounts = _inverse_scale_amount(X_test['amount'].values)
            flagged_amt = _real_amounts[y_pred.values == 1].sum()
        else:
            _real_amounts = None
            flagged_amt = 0
        flagged_X = X_test[y_pred==1]
        pattern_counts = {
            'Late Night':   int(flagged_X.get('is_night',       pd.Series(dtype=float)).sum()),
            'New Device':   int(flagged_X.get('is_new_device',  pd.Series(dtype=float)).sum()),
            'Round Amount': int(flagged_X.get('is_round_amount',pd.Series(dtype=float)).sum()),
            'Cross State':  int(flagged_X.get('cross_state',    pd.Series(dtype=float)).sum()),
        }
        if 'velocity_1h' in flagged_X.columns:
            pattern_counts['Hi Velocity'] = int((flagged_X['velocity_1h'] >= 4).sum())
        if 'cross_bank' in flagged_X.columns:
            pattern_counts['Cross Bank'] = int(flagged_X['cross_bank'].sum())
        pattern_counts = {k:v for k,v in pattern_counts.items() if v > 0}
        top_pattern = max(pattern_counts, key=pattern_counts.get) if pattern_counts else 'N/A'

    # ─── BLOOMBERG ───────────────────────────────────────────────────────────
    if _theme == 'bloomberg':
        _bb_css = """
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#0E131A;font-family:'Courier New',monospace;font-size:15px;color:#C0C8D0;}
.bb-wrap{background:#0E131A;color:#C0C8D0;font-family:'Courier New',monospace;}
.bb-topbar{background:#07090D;border-bottom:1px solid #1A2430;padding:10px 18px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.bb-topbar-logo{font-size:20px;font-weight:700;color:#FF8C00;letter-spacing:.12em;}
.bb-tickers{display:flex;gap:18px;margin-left:12px;font-size:13px;flex-wrap:wrap;}
.bb-tk-up{color:#00C896;} .bb-tk-dn{color:#E84040;} .bb-tk-lbl{color:#3A5060;}
.bb-topbar-time{margin-left:auto;font-size:12px;color:#2A3A48;}
.bb-body{display:flex;}
.bb-nav{width:170px;background:#07090D;border-right:1px solid #121C26;padding:10px 0;flex-shrink:0;}
.bb-nav-sect{font-size:11px;color:#FF8C00;letter-spacing:.18em;padding:8px 16px 4px;text-transform:uppercase;}
.bb-nav-item{padding:8px 16px;font-size:13px;color:#2A4050;cursor:pointer;border-left:2px solid transparent;letter-spacing:.04em;}
.bb-nav-item.a{color:#C0C8D0;border-left-color:#FF8C00;background:rgba(255,140,0,.06);}
.bb-main{flex:1;min-width:0;}
.bb-kpi-strip{display:grid;grid-template-columns:repeat(4,1fr);border-bottom:1px solid #121C26;}
.bb-kpi{padding:18px 20px;border-right:1px solid #121C26;}
.bb-kpi:last-child{border:none;}
.bb-kpi-lbl{font-size:11px;color:#2A4050;letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px;}
.bb-kpi-val{font-size:34px;font-weight:700;color:#C0C8D0;line-height:1;}
.bb-kpi-val.up{color:#00C896;} .bb-kpi-val.dn{color:#E84040;}
.bb-kpi-sub{font-size:12px;color:#1E3040;margin-top:5px;}
.bb-panels{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);}
.bb-panel{padding:16px;border-right:1px solid #121C26;}
.bb-panel:last-child{border:none;}
.bb-panel-hdr{font-size:12px;color:#FF8C00;letter-spacing:.14em;text-transform:uppercase;margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid #121C26;}
.bb-chart{display:flex;align-items:flex-end;gap:3px;height:80px;margin-bottom:14px;}
.bb-bar{flex:1;}
.bb-tbl{width:100%;border-collapse:collapse;}
.bb-th{font-size:11px;color:#1E3040;letter-spacing:.12em;text-transform:uppercase;padding:6px 8px;text-align:left;border-bottom:1px solid #121C26;}
.bb-td{font-size:14px;color:#7A9AAA;padding:7px 8px;border-bottom:1px solid #07090D;}
.bb-td.hi{color:#E84040;} .bb-td.lo{color:#00C896;}
.bb-badge-f{background:rgba(232,64,64,.14);border:1px solid rgba(232,64,64,.28);color:#E84040;font-size:11px;padding:2px 7px;letter-spacing:.06em;}
.bb-badge-n{background:rgba(0,200,150,.1);border:1px solid rgba(0,200,150,.2);color:#00C896;font-size:11px;padding:2px 7px;}
.bb-pat-row{display:flex;align-items:center;gap:10px;margin-bottom:10px;}
.bb-pat-lbl{font-size:13px;color:#8AABB8;width:110px;flex-shrink:0;letter-spacing:.04em;}
.bb-pat-bg{flex:1;height:10px;background:#121C26;}
.bb-pat-fill{height:10px;}
.bb-pat-num{font-size:14px;color:#E84040;width:48px;text-align:right;flex-shrink:0;font-weight:700;}
</style>
"""
        import datetime as _dt_mod
        _now = _dt_mod.datetime.now().strftime('%a %d %b %Y · %H:%M:%S').upper()
        if _has_data:
            _score_vals = scores[sc].values
            _bar_buckets = [_score_vals[i::12].mean() if len(_score_vals[i::12])>0 else 0 for i in range(12)]
            _bar_max = max(_bar_buckets) if max(_bar_buckets)>0 else 1
            _bars_html = ''.join(f'<div class="bb-bar" style="height:{max(int(bv/_bar_max*100),5)}%;background:{"#E84040" if bv>threshold else "#1A3040"};"></div>' for bv in _bar_buckets)
            _flagged_top = scores[scores[sc] >= threshold].head(4)
            _rows_html = ''.join(
                f'<tr><td class="bb-td">UPI-{idx_:05d}</td><td class="bb-td">{format_inr(_real_amounts[idx_]) if _real_amounts is not None else "—"}</td><td class="bb-td hi">{row_[sc]:.3f}</td><td class="bb-td"><span class="bb-badge-f">FRAUD</span></td></tr>'
                for idx_, row_ in _flagged_top.iterrows()
            ) or '<tr><td class="bb-td" colspan="4">No flagged transactions at current threshold</td></tr>'
            _pat_max = max(pattern_counts.values()) if pattern_counts else 1
            _pat_colors = ['#E84040','#FF8C00','#FF8C00','#00C896','#00C896','#00C896']
            _pat_html = ''.join(
                f'<div class="bb-pat-row"><span class="bb-pat-lbl">{pk}</span><div class="bb-pat-bg"><div class="bb-pat-fill" style="width:{int(pv/_pat_max*100)}%;background:{_pat_colors[pi%6]};"></div></div><span class="bb-pat-num" style="color:{_pat_colors[pi%6]};">{pv:,}</span></div>'
                for pi,(pk,pv) in enumerate(list(pattern_counts.items())[:5])
            )
            import streamlit.components.v1 as _components
            _components.html(f"""{_bb_css}
<div class="bb-wrap">
  <div class="bb-topbar">
    <span class="bb-topbar-logo">PAYGUARD</span>
    <div class="bb-tickers">
      <span><span class="bb-tk-lbl">FRAUD·RATE </span><span class="bb-tk-dn">{n_flagged/n_total*100:.2f}%</span></span>
      <span><span class="bb-tk-lbl">DATASET </span><span class="bb-tk-up">{dataset_total:,}</span></span>
      <span><span class="bb-tk-lbl">TRAIN </span><span class="bb-tk-up">{train_total:,}</span></span>
      <span><span class="bb-tk-lbl">TEST </span><span style="color:#FF8C00;">{test_total:,}</span></span>
      <span><span class="bb-tk-lbl">AT·RISK </span><span class="bb-tk-dn">{format_inr(flagged_amt)}</span></span>
      <span><span class="bb-tk-lbl">MODEL </span><span style="color:#FF8C00;">{model_choice[:2].upper()}:{threshold:.2f}</span></span>
    </div>
    <span class="bb-topbar-time">{_now}</span>
  </div>
  <div class="bb-body">
    <div class="bb-main" style="width:100%;margin-left:0;">
      <div class="bb-kpi-strip">
        <div class="bb-kpi"><div class="bb-kpi-lbl">Dataset</div><div class="bb-kpi-val">{dataset_total:,}</div><div class="bb-kpi-sub">All transactions</div></div>
        <div class="bb-kpi"><div class="bb-kpi-lbl">Training rows</div><div class="bb-kpi-val up">{train_total:,}</div><div class="bb-kpi-sub">Model fit split</div></div>
        <div class="bb-kpi"><div class="bb-kpi-lbl">Test evaluated</div><div class="bb-kpi-val">{test_total:,}</div><div class="bb-kpi-sub">{val_total:,} validation rows</div></div>
        <div class="bb-kpi"><div class="bb-kpi-lbl">Flagged</div><div class="bb-kpi-val dn">{n_flagged:,}</div><div class="bb-kpi-sub">{n_flagged/n_total*100:.2f}% test rate</div></div>
      </div>
      <div class="bb-panels">
        <div class="bb-panel">
          <div class="bb-panel-hdr">Score distribution — sample bins</div>
          <div class="bb-chart">{_bars_html}</div>
          <table class="bb-tbl">
            <tr><th class="bb-th">Txn ID</th><th class="bb-th">Amount</th><th class="bb-th">Score</th><th class="bb-th">Status</th></tr>
            {_rows_html}
          </table>
        </div>
        <div class="bb-panel">
          <div class="bb-panel-hdr">Fraud patterns</div>
          {_pat_html}
          <div style="margin-top:14px;font-size:13px;color:#1E3040;letter-spacing:.1em;">TOP PATTERN: <span style="color:#FF8C00;">{top_pattern.upper()}</span></div>
        </div>
      </div>
    </div>
  </div>
</div>
""", height=520, scrolling=False)
        else:
            st.markdown("<div style='color:#FF8C00;font-family:Courier New;font-size:11px;padding:16px;'>SETUP REQUIRED — run generate_upi_data.py → data_processing.py → models.py</div>", unsafe_allow_html=True)

    # ─── BROADSHEET ──────────────────────────────────────────────────────────
    elif _theme == 'broadsheet':
        _bs_css = """
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#F4EFE4;font-family:'Times New Roman',Times,serif;font-size:15px;color:#1A1A1A;}
.bs-wrap{background:#F4EFE4;color:#1A1A1A;font-family:'Times New Roman',Times,serif;}
.bs-mast{background:#1A1A1A;padding:14px 22px;display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;}
.bs-title{font-size:36px;color:#F4EFE4;font-style:italic;letter-spacing:-.02em;}
.bs-sub{font-size:12px;color:#888;letter-spacing:.22em;text-transform:uppercase;font-family:'Courier New',monospace;}
.bs-dateline{font-size:12px;color:#666;margin-left:auto;font-family:'Courier New',monospace;}
.bs-rule-thick{height:3px;background:#1A1A1A;}
.bs-rule-thin{height:1px;background:#C8C0A8;}
.bs-cols{display:grid;grid-template-columns:200px 1fr 180px;}
.bs-col{border-right:1px solid #C8C0A8;padding:18px 20px;}
.bs-col:last-child{border:none;}
.bs-col-hdr{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:#1A1A1A;font-family:'Courier New',monospace;border-bottom:1px solid #1A1A1A;padding-bottom:5px;margin-bottom:12px;}
.bs-stat{border:1.5px solid #1A1A1A;padding:12px;text-align:center;margin-bottom:12px;}
.bs-stat-n{font-size:34px;font-family:'Courier New',monospace;color:#1A1A1A;line-height:1;}
.bs-stat-l{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#666;margin-top:5px;font-family:'Courier New',monospace;}
.bs-headline{font-size:26px;line-height:1.08;color:#1A1A1A;margin-bottom:10px;}
.bs-headline em{font-style:normal;border-bottom:2px solid #1A1A1A;}
.bs-byline{font-size:12px;color:#777;font-style:italic;margin-bottom:10px;font-family:'Courier New',monospace;}
.bs-pull{font-size:18px;font-style:italic;border-top:2px solid #1A1A1A;border-bottom:2px solid #1A1A1A;padding:10px 0;margin:12px 0;line-height:1.3;}
.bs-body{font-size:14px;color:#333;line-height:1.65;}
.bs-txnrow{display:flex;justify-content:space-between;font-size:13px;font-family:'Courier New',monospace;padding:6px 0;border-bottom:1px solid #DDD8C8;}
.bs-flag{background:#1A1A1A;color:#F4EFE4;font-size:11px;padding:2px 7px;letter-spacing:.08em;}
.bs-bottom{display:grid;grid-template-columns:repeat(4,1fr);border-top:1.5px solid #1A1A1A;}
.bs-bot{padding:14px 18px;border-right:1px solid #C8C0A8;}
.bs-bot:last-child{border:none;}
.bs-bot-hdr{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:#888;font-family:'Courier New',monospace;margin-bottom:5px;}
.bs-bot-val{font-size:30px;font-family:'Courier New',monospace;color:#1A1A1A;line-height:1;}
.bs-bot-lbl{font-size:12px;color:#888;font-family:'Courier New',monospace;margin-top:3px;}
</style>
"""
        import datetime as _dt_mod
        _now_bs = _dt_mod.datetime.now().strftime('%a %d %b %Y · Ed. %j').upper()
        if _has_data:
            _flagged_idxs = scores[scores[sc] >= threshold].head(4).index.tolist()
            _txn_html = ''
            for i, idx_ in enumerate(_flagged_idxs):
                amt_disp = format_inr(_real_amounts[idx_]) if _real_amounts is not None else '—'
                sc_val = scores.loc[idx_, sc]
                _txn_html += f'<div class="bs-txnrow"><span>UPI-{idx_:05d}</span><span><span class="bs-flag">FRAUD</span></span></div><div class="bs-txnrow"><span>{amt_disp}</span><span>{sc_val:.3f}</span></div>'
                if i < len(_flagged_idxs)-1: _txn_html += '<div style="margin-top:4px;"></div>'
            if not _txn_html: _txn_html = '<div class="bs-txnrow"><span>No flags at threshold</span><span>—</span></div>'
            _pat_items = list(pattern_counts.items())[:4]
            _bot_html = ''.join(f'<div class="bs-bot"><div class="bs-bot-hdr">{pk}</div><div class="bs-bot-val">{pv:,}</div><div class="bs-bot-lbl">{pv/n_flagged*100 if n_flagged else 0:.1f}% of flags</div></div>' for pk,pv in _pat_items)
            while len(_pat_items) < 4: _bot_html += '<div class="bs-bot"></div>'; _pat_items.append(None)
            import streamlit.components.v1 as _components
            _components.html(f"""{_bs_css}
<div class="bs-wrap">
  <div class="bs-mast">
    <div class="bs-title">PayGuard</div>
    <div class="bs-sub">Fraud Intelligence Daily</div>
    <div class="bs-dateline">{_now_bs}</div>
  </div>
  <div class="bs-rule-thick"></div>
  <div class="bs-cols">
    <div class="bs-col">
      <div class="bs-col-hdr">Today's verdict</div>
      <div class="bs-stat"><div class="bs-stat-n">{dataset_total:,}</div><div class="bs-stat-l">Dataset</div></div>
      <div class="bs-stat"><div class="bs-stat-n">{train_total:,}</div><div class="bs-stat-l">Train rows</div></div>
      <div class="bs-stat"><div class="bs-stat-n">{n_flagged:,}</div><div class="bs-stat-l">Flagged</div></div>
      <div class="bs-stat"><div class="bs-stat-n">{n_flagged/n_total*100:.2f}%</div><div class="bs-stat-l">Flag rate</div></div>
    </div>
    <div class="bs-col">
      <div class="bs-col-hdr">Lead story</div>
      <div class="bs-headline">Pattern analysis reveals <em>{top_pattern.lower()}</em> as top fraud driver</div>
      <div class="bs-byline">PayGuard Anomaly Engine · {model_choice} · Threshold {threshold:.2f}</div>
      <div class="bs-pull">"Of {n_total:,} transactions evaluated, 1 in {int(n_total/max(n_flagged,1))} was flagged — {top_pattern.lower()} leads all patterns."</div>
      <div class="bs-body">Dataset coverage is {dataset_total:,} total transactions: {train_total:,} train, {val_total:,} validation, and {test_total:,} test. The {model_choice} model flagged {n_flagged:,} anomalous test transactions ({n_flagged/n_total*100:.2f}% flag rate) with {format_inr(flagged_amt)} in total exposure.</div>
    </div>
    <div class="bs-col">
      <div class="bs-col-hdr">Flagged log</div>
      {_txn_html}
    </div>
  </div>
  <div class="bs-rule-thin"></div>
  <div class="bs-bottom">{_bot_html}</div>
</div>
""", height=520, scrolling=False)
        else:
            st.markdown("<div style='font-family:Courier New;font-size:11px;color:#666;padding:16px;'>Setup required — run pipeline first.</div>", unsafe_allow_html=True)

    # ─── SWISS ───────────────────────────────────────────────────────────────
    elif _theme == 'swiss':
        _sw_css = """
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{background:#FFF;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;font-size:15px;color:#111;}
.sw-wrap{background:#FFFFFF;color:#111;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;}
.sw-header{background:#111;padding:14px 24px;display:flex;align-items:center;flex-wrap:wrap;}
.sw-logo{font-size:24px;font-weight:700;color:#FFF;letter-spacing:.28em;text-transform:uppercase;}
.sw-logo-accent{color:#E8320A;}
.sw-header-nav{margin-left:auto;display:flex;flex-wrap:wrap;}
.sw-hn{font-size:12px;color:#555;letter-spacing:.18em;text-transform:uppercase;padding:0 16px;border-left:1px solid #333;}
.sw-hn.a{color:#FFF;}
.sw-rule-r{height:4px;background:#E8320A;}
.sw-body{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,3fr);}
.sw-left{border-right:1px solid #E0E0E0;padding:24px 28px;}
.sw-right{padding:24px 28px;}
.sw-overline{font-size:12px;letter-spacing:.28em;text-transform:uppercase;color:#E8320A;margin-bottom:10px;font-weight:700;}
.sw-headline{font-size:42px;font-weight:700;line-height:1.0;color:#111;margin-bottom:16px;letter-spacing:-.02em;}
.sw-stat-row{display:flex;border-top:1px solid #E0E0E0;border-bottom:1px solid #E0E0E0;margin-bottom:18px;}
.sw-stat{flex:1;padding:16px 0;border-right:1px solid #E0E0E0;}
.sw-stat:last-child{border:none;}
.sw-stat-n{font-size:34px;font-weight:700;color:#111;line-height:1;}
.sw-stat-n.red{color:#E8320A;}
.sw-stat-l{font-size:12px;letter-spacing:.18em;text-transform:uppercase;color:#999;margin-top:5px;}
.sw-sect-hdr{font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:#999;border-bottom:1px solid #E0E0E0;padding-bottom:6px;margin-bottom:12px;font-weight:700;}
.sw-txnrow{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #F0F0F0;font-size:15px;}
.sw-txn-id{font-family:'Courier New',monospace;color:#888;font-size:13px;}
.sw-txn-amt{font-weight:700;color:#111;}
.sw-txn-flag{font-size:12px;font-weight:700;letter-spacing:.1em;background:#E8320A;color:#FFF;padding:3px 10px;}
.sw-txn-ok{font-size:12px;letter-spacing:.1em;color:#999;border:1px solid #DDD;padding:3px 10px;}
.sw-pat-row{display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.sw-pat-lbl{font-size:14px;font-weight:700;color:#111;width:120px;flex-shrink:0;}
.sw-pat-bg{flex:1;height:12px;background:#F0F0F0;}
.sw-pat-fill{height:12px;background:#E8320A;}
.sw-pat-num{font-size:15px;font-weight:700;color:#E8320A;width:52px;text-align:right;flex-shrink:0;font-family:'Courier New',monospace;}
.sw-bottom{display:grid;grid-template-columns:repeat(4,1fr);border-top:3px solid #111;}
.sw-bot{padding:16px 20px;border-right:1px solid #E0E0E0;}
.sw-bot:last-child{border:none;}
.sw-bot-hdr{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:#999;margin-bottom:5px;font-weight:700;}
.sw-bot-val{font-size:32px;font-weight:700;color:#111;line-height:1;font-family:'Courier New',monospace;}
.sw-bot-lbl{font-size:12px;color:#999;letter-spacing:.06em;margin-top:3px;}
</style>
"""
        if _has_data:
            _flagged_idxs_sw = scores[scores[sc] >= threshold].head(5).index.tolist()
            _normal_idxs_sw  = scores[scores[sc] < threshold].head(2).index.tolist()
            _txn_rows_sw = ''.join(
                f'<div class="sw-txnrow"><span class="sw-txn-id">UPI-{idx_:05d}</span><span class="sw-txn-amt">{format_inr(_real_amounts[idx_]) if _real_amounts is not None else "—"}</span><span class="sw-txn-flag">FRAUD</span></div>'
                for idx_ in _flagged_idxs_sw
            ) + ''.join(
                f'<div class="sw-txnrow"><span class="sw-txn-id">UPI-{idx_:05d}</span><span class="sw-txn-amt">{format_inr(_real_amounts[idx_]) if _real_amounts is not None else "—"}</span><span class="sw-txn-ok">normal</span></div>'
                for idx_ in _normal_idxs_sw
            ) or '<div class="sw-txnrow"><span class="sw-txn-id">No data</span><span>—</span></div>'
            _pat_max_sw = max(pattern_counts.values()) if pattern_counts else 1
            _pat_html_sw = ''.join(
                f'<div class="sw-pat-row"><span class="sw-pat-lbl">{pk}</span><div class="sw-pat-bg"><div class="sw-pat-fill" style="width:{int(pv/_pat_max_sw*100)}%;"></div></div><span class="sw-pat-num">{pv:,}</span></div>'
                for pk,pv in list(pattern_counts.items())[:4]
            )
            _bot_items_sw = list(pattern_counts.items())[:4]
            _bot_html_sw = ''.join(f'<div class="sw-bot"><div class="sw-bot-hdr">{pk}</div><div class="sw-bot-val">{pv:,}</div><div class="sw-bot-lbl">{pv/n_flagged*100 if n_flagged else 0:.1f}% of flags</div></div>' for pk,pv in _bot_items_sw)
            while len(_bot_items_sw) < 4: _bot_html_sw += '<div class="sw-bot"></div>'; _bot_items_sw.append(None)
            import streamlit.components.v1 as _components
            _components.html(f"""{_sw_css}
<div class="sw-wrap">
  <div class="sw-header">
    <div class="sw-logo">Pay<span class="sw-logo-accent">Guard</span></div>
    <div class="sw-header-nav">
      <span class="sw-hn a">Overview</span><span class="sw-hn">Live score</span><span class="sw-hn">Scan</span><span class="sw-hn">Analytics</span>
    </div>
  </div>
  <div class="sw-rule-r"></div>
  <div class="sw-body">
    <div class="sw-left">
      <div class="sw-overline">Real-time UPI monitoring</div>
      <div class="sw-headline">{n_flagged:,} flagged<br>today — {top_pattern.lower()}<br>leads patterns</div>
      <div class="sw-stat-row">
        <div class="sw-stat"><div class="sw-stat-n">{test_total:,}</div><div class="sw-stat-l">Test eval</div></div>
        <div class="sw-stat"><div class="sw-stat-n red">{n_flagged/n_total*100:.1f}%</div><div class="sw-stat-l">Flag rate</div></div>
        <div class="sw-stat"><div class="sw-stat-n">{val_total:,}</div><div class="sw-stat-l">Validation</div></div>
      </div>
      <div class="sw-sect-hdr">Pattern breakdown</div>
      {_pat_html_sw}
    </div>
    <div class="sw-right">
      <div class="sw-sect-hdr">Recent transactions</div>
      {_txn_rows_sw}
      <div style="margin-top:14px;">
        <div class="sw-sect-hdr">Model status</div>
        <div class="sw-txnrow"><span style="font-size:14px;color:#999;font-family:'Courier New',monospace;">Model</span><span style="font-size:14px;font-weight:700;">{model_choice}</span></div>
        <div class="sw-txnrow"><span style="font-size:14px;color:#999;font-family:'Courier New',monospace;">Threshold</span><span style="font-size:14px;font-weight:700;">{threshold:.2f}</span></div>
        <div class="sw-txnrow"><span style="font-size:14px;color:#999;font-family:'Courier New',monospace;">Dataset</span><span style="font-size:14px;font-weight:700;">{dataset_total:,} txns</span></div>
        <div class="sw-txnrow"><span style="font-size:14px;color:#999;font-family:'Courier New',monospace;">Train / Val / Test</span><span style="font-size:14px;font-weight:700;">{train_total:,} / {val_total:,} / {test_total:,}</span></div>
        <div class="sw-txnrow" style="border:none;"><span style="font-size:14px;color:#999;font-family:'Courier New',monospace;">Top pattern</span><span style="font-size:14px;font-weight:700;color:#E8320A;">{top_pattern}</span></div>
      </div>
    </div>
  </div>
  <div class="sw-bottom">
    <div class="sw-bot"><div class="sw-bot-hdr">Dataset</div><div class="sw-bot-val">{dataset_total:,}</div><div class="sw-bot-lbl">Total rows</div></div>
    <div class="sw-bot"><div class="sw-bot-hdr">Training</div><div class="sw-bot-val">{train_total:,}</div><div class="sw-bot-lbl">Model fit</div></div>
    <div class="sw-bot"><div class="sw-bot-hdr">Test eval</div><div class="sw-bot-val">{test_total:,}</div><div class="sw-bot-lbl">{val_total:,} validation</div></div>
    <div class="sw-bot"><div class="sw-bot-hdr">Flagged</div><div class="sw-bot-val" style="color:#E8320A;">{n_flagged:,}</div><div class="sw-bot-lbl">{n_flagged/n_total*100:.2f}% rate</div></div>
    <div class="sw-bot"><div class="sw-bot-hdr">Top pattern</div><div class="sw-bot-val" style="font-size:22px;margin-top:3px;">{top_pattern.split()[0] if top_pattern!='N/A' else '—'}</div><div class="sw-bot-lbl">{pattern_counts.get(top_pattern,0):,} flags</div></div>
  </div>
</div>
""", height=520, scrolling=False)
        else:
            st.markdown("<div style='padding:20px;font-size:11px;color:#999;'>Setup required — run pipeline first.</div>", unsafe_allow_html=True)

    # ─── ENRICHED OVERVIEW CHARTS (all themes, below the iframe) ─────────────
    if _has_data:
        st.divider()

        # Row 1: Score distribution + Fraud by hour
        _ov1, _ov2 = st.columns(2)
        _ens = scores['ensemble_score'] if 'ensemble_score' in scores.columns else scores[sc]
        _fraud_mask = (y_true == 1)

        with _ov1:
            st.markdown("<p class='section-header'>Anomaly Score Distribution</p>", unsafe_allow_html=True)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=_ens[~_fraud_mask], name='Normal',           nbinsx=60, opacity=0.75, marker_color='#4C3EE8' if _theme=='swiss' else ('#00C896' if _theme=='bloomberg' else '#1A1A1A')))
            fig_dist.add_trace(go.Histogram(x=_ens[_fraud_mask],  name='Fraud (gt label)', nbinsx=60, opacity=0.85, marker_color='#E84040' if _theme=='bloomberg' else '#E8320A' if _theme=='swiss' else '#1A1A1A'))
            fig_dist.add_vline(x=threshold, line_dash='dash',
                               line_color='#FF8C00' if _theme=='bloomberg' else '#E8320A',
                               annotation_text=f'Threshold {threshold}',
                               annotation_font_color='#FF8C00' if _theme=='bloomberg' else '#E8320A')
            fig_dist.update_layout(barmode='overlay', legend=dict(orientation='h',y=1.1), margin=dict(l=0,r=0,t=10,b=0), xaxis_title='Score', yaxis_title='Count')
            st.plotly_chart(light_chart(fig_dist, 280), use_container_width=True)

        with _ov2:
            st.markdown("<p class='section-header'>Fraud Rate by Hour of Day</p>", unsafe_allow_html=True)
            if 'txn_hour' in X_test.columns:
                _hrs = X_test['txn_hour'].values
                _fs2 = load_feature_scaler()
                if _fs2 is not None:
                    try:
                        _hr_idx = list(X_test.columns).index('txn_hour')
                        _hrs_real = (_hrs * _fs2.scale_[_hr_idx] + _fs2.mean_[_hr_idx]).round().astype(int) % 24
                    except Exception:
                        _hrs_real = (_hrs * 23).round().astype(int) % 24
                else:
                    _hrs_real = (_hrs * 23).round().astype(int) % 24
                _hr_df  = pd.DataFrame({'hour': _hrs_real, 'flagged': y_pred})
                _hr_grp = _hr_df.groupby('hour').agg(total=('flagged','count'), flagged=('flagged','sum')).reset_index()
                _hr_grp['rate'] = _hr_grp['flagged'] / _hr_grp['total'].clip(lower=1)
                _accent = '#FF8C00' if _theme=='bloomberg' else '#E8320A'
                fig_hr = px.bar(_hr_grp, x='hour', y='rate',
                                color='rate', color_continuous_scale=[[0,'#1A3040' if _theme=='bloomberg' else '#E8E8E8'],[1,_accent]],
                                labels={'hour':'Hour (24h)','rate':'Flag Rate'})
                fig_hr.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(light_chart(fig_hr, 280), use_container_width=True)
            else:
                st.info("txn_hour not in X_test.")

        # Row 2: Amount buckets + Model AUC comparison
        _ov3, _ov4 = st.columns(2)

        with _ov3:
            st.markdown("<p class='section-header'>Flagged Transactions by Amount Bucket</p>", unsafe_allow_html=True)
            if _real_amounts is not None:
                _bins   = [0,500,2000,10000,50000,100000,float('inf')]
                _labels = ['<₹500','₹500–2k','₹2k–10k','₹10k–50k','₹50k–1L','>₹1L']
                _bucketed = pd.cut(_real_amounts[y_pred==1], bins=_bins, labels=_labels)
                _counts   = _bucketed.value_counts().reindex(_labels, fill_value=0)
                _bar_color = '#FF8C00' if _theme=='bloomberg' else ('#1A1A1A' if _theme=='broadsheet' else '#E8320A')
                fig_amt = px.bar(x=_labels, y=_counts.values,
                                 color=_counts.values,
                                 color_continuous_scale=[[0,'#1A3040' if _theme=='bloomberg' else '#E8E8E8'],[1,_bar_color]],
                                 labels={'x':'Amount Range','y':'Flagged Count'})
                fig_amt.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(light_chart(fig_amt, 260), use_container_width=True)

        with _ov4:
            st.markdown("<p class='section-header'>Model ROC-AUC Comparison</p>", unsafe_allow_html=True)
            try:
                from sklearn.metrics import roc_auc_score as _ras
                _auc_data = {mn: round(_ras(y_true, scores[sc2]), 4) for mn,sc2 in SCORE_MAP.items() if sc2 in scores.columns}
                if _auc_data:
                    _best = max(_auc_data.values())
                    _accent = '#FF8C00' if _theme=='bloomberg' else ('#1A1A1A' if _theme=='broadsheet' else '#E8320A')
                    _norm   = '#1A3040' if _theme=='bloomberg' else ('#888' if _theme=='broadsheet' else '#888')
                    fig_auc = go.Figure(go.Bar(
                        x=list(_auc_data.values()), y=list(_auc_data.keys()), orientation='h',
                        marker_color=[_accent if v==_best else _norm for v in _auc_data.values()],
                        text=[f'{v:.4f}' for v in _auc_data.values()], textposition='outside'
                    ))
                    fig_auc.update_layout(xaxis_range=[0.5,1.0], margin=dict(l=0,r=60,t=10,b=0), xaxis_title='ROC-AUC')
                    st.plotly_chart(light_chart(fig_auc, 260), use_container_width=True)
            except ImportError:
                st.info("sklearn not available.")

        # Score percentile strip
        st.divider()
        _p50,_p90,_p95,_p99 = float(_ens.quantile(.50)), float(_ens.quantile(.90)), float(_ens.quantile(.95)), float(_ens.quantile(.99))
        _pc1,_pc2,_pc3,_pc4,_pc5 = st.columns(5)
        _pc1.markdown("<p class='section-header' style='padding-top:12px'>Score Percentiles</p>", unsafe_allow_html=True)
        _pc2.metric("P50 (median)", f"{_p50:.3f}")
        _pc3.metric("P90", f"{_p90:.3f}")
        _pc4.metric("P95", f"{_p95:.3f}")
        _pc5.metric("P99", f"{_p99:.3f}")

    else:
        st.markdown("""
<div style='background:#F7F8FC;border:1px solid #E5E7EB;border-radius:12px;padding:20px 24px;'>
<p style='color:#6B7280;font-size:0.8rem;text-transform:uppercase;letter-spacing:.1em;margin:0 0 12px'>Setup required — run in order</p>
<pre style='color:#0F1117;font-size:0.85rem;line-height:1.8;margin:0'>
1. python src/generate_upi_data.py
2. python src/data_processing.py
3. python src/models.py
4. streamlit run app.py
</pre>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SCAN
# ════════════════════════════════════════════════════════════════════════════
elif page == '📤 Scan':
    st.markdown("""
    <div class='hero-banner' style='padding:28px 40px'>
        <h1 style='font-size:1.8rem !important'>📤 Batch Scanner</h1>
        <p>Upload raw UPI transactions or pre-processed features for bulk scoring</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Download CSV template (raw UPI format)"):
        sample = pd.DataFrame([{'sender_upi':'user1234@okicici','receiver_upi':'merchant5678@ybl','amount':85000,'timestamp':'2024-03-15 02:34:00','upi_app':'GPay','merchant_category':'p2p','sender_bank':'ICICI','receiver_bank':'SBI','sender_state':'Delhi','receiver_state':'Maharashtra'}])
        st.dataframe(sample, use_container_width=True)
        st.download_button('⬇ Download template', sample.to_csv(index=False), file_name='upi_template.csv', mime='text/csv')

    uploaded = st.file_uploader('Drop a CSV here', type='csv')
    if not uploaded:
        if os.path.exists(X_TEST_PATH):
            if st.button('▶ Run sample test dataset'):
                st.session_state['use_sample_data'] = True
                st.rerun()
        st.info('Upload a CSV above, or run the bundled sample dataset to test with pre-computed data.')

    sample_data_mode = st.session_state.get('use_sample_data', False)
    source_df = None
    if uploaded:
        source_df = pd.read_csv(uploaded); sample_data_mode = False; st.session_state['use_sample_data'] = False
    elif sample_data_mode and os.path.exists(X_TEST_PATH):
        source_df = pd.read_csv(X_TEST_PATH)

    if source_df is not None:
        raw_mode = is_raw_upi(source_df)
        st.markdown(f"**{len(source_df):,} rows** · {'Raw UPI format — encoding automatically' if raw_mode else 'Pre-processed features'}")
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(source_df.head(5), use_container_width=True)
        with st.spinner(f'Encoding features & running {model_choice}...'):
            try:
                feature_df = encode_raw_df(source_df) if raw_mode else source_df.copy()
                result = cached_inference(feature_df.to_json(double_precision=15), model_name=model_choice, threshold=threshold)
                display_df = source_df.copy()
                display_df['anomaly_score'] = result['anomaly_score'].values
                display_df['is_fraud']      = result['is_fraud'].values
                n_fraud = int(display_df['is_fraud'].sum())
                n_total = len(display_df)
                flagged_amt = display_df[display_df['is_fraud']]['amount'].sum() if 'amount' in display_df.columns else 0
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Total Transactions', f'{n_total:,}')
                c2.metric('Flagged as Fraud', f'{n_fraud:,}')
                c3.metric('Flag Rate', f'{n_fraud/n_total*100:.2f}%')
                c4.metric('Amount at Risk', format_inr(flagged_amt))
                st.markdown("#### Model Comparison (same data, same threshold)")
                comp_cols = st.columns(3)
                for i, mn in enumerate(['Isolation Forest','Autoencoder','LOF']):
                    try:
                        r2 = cached_inference(feature_df.to_json(double_precision=15), model_name=mn, threshold=threshold)
                        nf = int(r2['is_fraud'].sum())
                        comp_cols[i].metric(mn, f'{nf:,} flagged', f"{nf/n_total*100:.2f}%")
                    except Exception as e:
                        comp_cols[i].metric(mn, 'Error')
                st.divider()
                fraud_df = display_df[display_df['is_fraud']].copy()
                fraud_df['fraud_reasons'] = fraud_df.apply(lambda r: ' '.join(get_fraud_tags(r)), axis=1)
                tab1, tab2 = st.tabs([f'🚨 Flagged ({n_fraud})', f'✅ Normal ({n_total-n_fraud})'])
                with tab1:
                    if n_fraud == 0:
                        st.success('No transactions flagged. Try lowering the threshold.')
                    else:
                        show_cols = [c for c in ['sender_upi','receiver_upi','amount','upi_app','merchant_category','txn_hour','anomaly_score','fraud_reasons'] if c in fraud_df.columns]
                        sorted_fraud = fraud_df[show_cols].sort_values('anomaly_score', ascending=False)
                        if len(sorted_fraud) > 200:
                            st.caption(f'Showing top 200 of {len(sorted_fraud):,}. Download for full list.')
                        st.dataframe(sorted_fraud.head(200), use_container_width=True)
                        st.download_button('⬇ Download Flagged', fraud_df.to_csv(index=False), file_name='flagged_transactions.csv', mime='text/csv')
                        log_audit(page='Scan', model=model_choice, threshold=threshold, input_data=f"csv_{len(display_df)}_rows", n_rows=len(display_df), n_flagged=len(fraud_df), top_score=float(display_df['anomaly_score'].max()) if 'anomaly_score' in display_df.columns else 0.0, verdict=f"{len(fraud_df)} flagged of {len(display_df)}")
                with tab2:
                    normal_df  = display_df[~display_df['is_fraud']]
                    show_cols2 = [c for c in ['sender_upi','receiver_upi','amount','upi_app','merchant_category','anomaly_score'] if c in normal_df.columns]
                    st.dataframe(normal_df[show_cols2] if show_cols2 else normal_df, use_container_width=True)
            except FileNotFoundError as e:
                st.error(f'Model not found: {e}')
            except Exception as e:
                st.error(f'Error: {e}')
                import traceback; st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
elif page == '📊 Analytics':
    st.markdown("""
    <div class='hero-banner' style='padding:28px 40px'>
        <h1 style='font-size:1.8rem !important'>📊 Analytics</h1>
        <p>Model performance, explainability, and visualization suite</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(ANOMALY_SCORES_PATH) and os.path.exists(Y_TEST_PATH):
        from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_score, recall_score, f1_score)
        scores = pd.read_csv(ANOMALY_SCORES_PATH)
        y_true = pd.read_csv(Y_TEST_PATH).values.ravel()
        _n = min(len(scores), len(y_true))
        scores = scores.iloc[:_n].reset_index(drop=True)
        y_true = y_true[:_n]

        atab1, atab2, atab3 = st.tabs(['📈 Model Performance', '💡 SHAP & Explainability', '🗺️ Charts'])

        with atab1:
            st.markdown("### Model Comparison")
            rows = []
            for name, col in SCORE_MAP.items():
                if col in scores.columns:
                    yp = (scores[col] >= threshold).astype(int)
                    try: auc = roc_auc_score(y_true, scores[col])
                    except: auc = 0
                    rows.append({'Model':name,'Flagged':int(yp.sum()),'Flag Rate':f"{yp.mean()*100:.2f}%",'Precision':f"{precision_score(y_true,yp,zero_division=0):.3f}",'Recall':f"{recall_score(y_true,yp,zero_division=0):.3f}",'F1':f"{f1_score(y_true,yp,zero_division=0):.3f}",'ROC-AUC':f"{auc:.4f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.divider()
            st.markdown("### ROC Curves")
            fig_roc = go.Figure()
            for name, col in SCORE_MAP.items():
                if col in scores.columns:
                    try:
                        fpr, tpr, _ = roc_curve(y_true, scores[col])
                        auc = roc_auc_score(y_true, scores[col])
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={auc:.3f})', line=dict(color=COLORS.get(name,'#4C3EE8'), width=2)))
                    except Exception as e:
                        log.warning(f"ROC failed for {name}: {e}")
            fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random',line=dict(color='#888',dash='dash',width=1)))
            fig_roc.update_layout(xaxis_title='FPR', yaxis_title='TPR')
            st.plotly_chart(light_chart(fig_roc, 420), use_container_width=True)
            st.divider()
            st.markdown("### Threshold Simulator")
            sim_model = st.selectbox('Model', list(SCORE_MAP.keys()), key='sim_sel')
            sim_col   = SCORE_MAP[sim_model]
            if sim_col in scores.columns:
                thresholds = np.linspace(0.05, 0.99, 90)
                precs,recs,f1s,frates = [],[],[],[]
                for t in thresholds:
                    yp = (scores[sim_col] >= t).astype(int)
                    p  = precision_score(y_true, yp, zero_division=0)
                    r  = recall_score(y_true, yp, zero_division=0)
                    precs.append(p); recs.append(r); f1s.append(2*p*r/(p+r+1e-8)); frates.append(yp.mean())
                fig_sim = go.Figure()
                fig_sim.add_trace(go.Scatter(x=thresholds, y=precs,  name='Precision', line=dict(color='#4CC9F0',width=2)))
                fig_sim.add_trace(go.Scatter(x=thresholds, y=recs,   name='Recall',    line=dict(color='#E03E3E',width=2)))
                fig_sim.add_trace(go.Scatter(x=thresholds, y=f1s,    name='F1',        line=dict(color='#F7B731',width=2)))
                fig_sim.add_trace(go.Scatter(x=thresholds, y=frates, name='Flag Rate', line=dict(color='#888',dash='dot',width=1)))
                _ann_col = '#FF8C00' if _theme=='bloomberg' else ('#1A1A1A' if _theme=='broadsheet' else '#E8320A')
                fig_sim.add_vline(x=threshold, line_dash='dash', line_color=_ann_col, annotation_text=f'Current: {threshold}', annotation_font_color=_ann_col)
                fig_sim.update_layout(xaxis_title='Threshold', yaxis_title='Score', yaxis=dict(range=[0,1]))
                st.plotly_chart(light_chart(fig_sim, 350), use_container_width=True)
                yp_cur = (scores[sim_col] >= threshold).astype(int)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('Flags', f'{yp_cur.sum():,}')
                c2.metric('Precision', f"{precision_score(y_true,yp_cur,zero_division=0):.3f}")
                c3.metric('Recall',    f"{recall_score(y_true,yp_cur,zero_division=0):.3f}")
                c4.metric('Flag Rate', f"{yp_cur.mean()*100:.2f}%")

        with atab2:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### SHAP Feature Importance")
                shap_path = os.path.join(CHARTS_DIR, 'shap_summary.png')
                if os.path.exists(shap_path):
                    st.image(shap_path, use_container_width=True)
                else:
                    st.info('Run analysis.py + visualization.py')
            with col_r:
                st.markdown("### Confusion Matrix")
                cm_path = os.path.join(CHARTS_DIR, 'confusion_matrix.png')
                if os.path.exists(cm_path):
                    st.image(cm_path, use_container_width=True)
                else:
                    st.info('Run visualization.py')

        with atab3:
            chart_files = {
                'PCA Projection': ('pca_plot.html','All transactions compressed to 2D.'),
                't-SNE Clusters': ('tsne_plot.html','t-SNE cluster view.'),
                'Score Distribution': ('score_dist.html','Anomaly score histograms.'),
            }
            for title, (fname, caption) in chart_files.items():
                path = os.path.join(CHARTS_DIR, fname)
                st.markdown(f"### {title}")
                st.caption(caption)
                if os.path.exists(path):
                    load_key = f'load_{fname}'
                    col_btn, col_dl = st.columns([4,1])
                    with col_btn:
                        if st.session_state.get(load_key):
                            with open(path,'r',encoding='utf-8') as f: st.components.v1.html(f.read(), height=500, scrolling=False)
                            if st.button(f'Hide {title}', key=f'hide_{fname}'): st.session_state[load_key]=False; st.rerun()
                        else:
                            if st.button(f'▶ Load {title}  ({os.path.getsize(path)/1_048_576:.1f} MB)', key=f'btn_{fname}'): st.session_state[load_key]=True; st.rerun()
                    with col_dl:
                        with open(path,'rb') as f_dl: st.download_button('⬇', f_dl.read(), file_name=fname, mime='text/html', key=f'dl_{fname}')
                else:
                    st.info('Run `python src/visualization.py` to generate.')
                st.divider()
    else:
        st.warning('Run `python src/models.py` first.')


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE SCORE
# ════════════════════════════════════════════════════════════════════════════
elif page == '📡 Live Score':
    from datetime import datetime as _dt
    import urllib.parse as _urlparse

    st.markdown("""
    <div class='hero-banner' style='padding:28px 40px'>
        <h1 style='font-size:1.8rem !important'>📡 Live Transaction Scorer</h1>
        <p>Score a single UPI transaction instantly</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("How Live Score works and how to test it"):
        st.markdown("""
Live Score builds one transaction row from the form, encodes it with the same feature pipeline as batch scanning, then runs Isolation Forest, Autoencoder, LOF, and the weighted Ensemble. The verdict comes from the Ensemble score crossing the active anomaly threshold, while the Risk Summary explains the score with visible signals such as amount, velocity, new device, night timing, cross-state, and cross-bank movement.

For screenshots, use the quick-fill scenarios, click **Score this transaction**, and capture the verdict banner, model comparison, fraud signals, risk summary, and gauge. Good presentation checks are **Normal Transaction** for a low-risk baseline, **Late Night Fraud** for a high-score case, **High Velocity** for behavior-based risk, and **Cross-State New Device** for device/location risk. The receipt sample generator creates a synthetic receipt card only for the analysis workflow; it does not submit or persist a real payment.
        """)

    with st.expander("📷 Auto-fill from UPI payment screenshot (OCR)"):
        st.markdown("""
        <div class='ocr-panel'>
            <div class='ocr-panel__title'>Screenshot Autofill</div>
            <div class='ocr-panel__subtitle'>Upload a GPay, PhonePe, or Paytm receipt image. The widget extracts the amount and UPI ID when OCR is available.</div>
            <div class='ocr-panel__meta'>
                <span class='ocr-chip ocr-chip--accent'>PNG / JPG</span>
                <span class='ocr-chip'>Max 5 MB</span>
                <span class='ocr-chip ocr-chip--muted'>Optional autofill</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded_img = st.file_uploader("Upload payment screenshot", type=["png","jpg","jpeg"], key="ocr_img")
        # Reset processed flag when a new image is uploaded
        _ocr_file_id = uploaded_img.file_id if uploaded_img else None
        if _ocr_file_id != st.session_state.get('_ocr_last_file_id'):
            st.session_state['_ocr_last_file_id'] = _ocr_file_id
            st.session_state.pop('_ocr_processed', None)
        # Re-display cached OCR results on subsequent renders (after rerun)
        if uploaded_img and st.session_state.get('_ocr_processed') and '_ocr_summary_html' in st.session_state:
            st.markdown(st.session_state['_ocr_summary_html'], unsafe_allow_html=True)

        if uploaded_img and not st.session_state.get('_ocr_processed'):
            st.session_state['_ocr_processed'] = True
            if uploaded_img.size > 5*1024*1024:
                st.markdown("<div class='ocr-status ocr-status--warning'>File too large. Max 5 MB.</div>", unsafe_allow_html=True)
            else:
                try:
                    from PIL import Image
                    import pytesseract, re
                    tesseract_cmd = _configure_tesseract_binary()
                    if not tesseract_cmd:
                        st.markdown("<div class='ocr-status ocr-status--info'>OCR is available only when the Tesseract binary is installed. Install it or set TESSERACT_CMD to the executable path, then upload the screenshot again.</div>", unsafe_allow_html=True)
                    else:
                        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                        img = Image.open(uploaded_img)
                        if img.width > 4096 or img.height > 4096:
                            st.markdown("<div class='ocr-status ocr-status--warning'>Image too large. Max 4096×4096.</div>", unsafe_allow_html=True)
                        else:
                            text = pytesseract.image_to_string(img)
                            normalized_text = text.replace('₹', '₹ ').replace('Rs.', 'Rs ').replace('Rs', 'Rs ').replace('INR', 'INR ')

                            def _parse_amount_from_ocr(raw_text):
                                amount_patterns = [
                                    r'(?:₹|Rs\.?|INR)\s*([\d,]+(?:\.\d{1,2})?)',
                                    r'amount\s*[:\-]?\s*([\d,]+(?:\.\d{1,2})?)',
                                    r'credited\s*[:\-]?\s*([\d,]+(?:\.\d{1,2})?)',
                                    r'received\s*[:\-]?\s*([\d,]+(?:\.\d{1,2})?)',
                                ]
                                for pattern in amount_patterns:
                                    for match in re.finditer(pattern, raw_text, re.IGNORECASE):
                                        candidate = match.group(1).replace(',', '')
                                        try:
                                            value = float(candidate)
                                        except ValueError:
                                            continue
                                        if 0 < value <= UPI_MAX_LIMIT:
                                            return value

                                candidates = []
                                for match in re.finditer(r'\b\d[\d,]*(?:\.\d{1,2})?\b', raw_text):
                                    candidate = match.group(0).replace(',', '')
                                    try:
                                        value = float(candidate)
                                    except ValueError:
                                        continue
                                    if 0 < value <= UPI_MAX_LIMIT:
                                        has_decimal = '.' in match.group(0)
                                        candidates.append((has_decimal, value))
                                if candidates:
                                    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                                    return candidates[0][1]
                                return None

                            parsed_amount = _parse_amount_from_ocr(normalized_text)
                            upi_matches = re.findall(r'[\w.\-]+@[\w]+', text)
                            parsed_sender_upi = upi_matches[0] if upi_matches else None
                            parsed_receiver_upi = upi_matches[-1] if len(upi_matches) > 1 else parsed_sender_upi
                            parsed_txn_hour = None
                            parsed_txn_day = None
                            parsed_txn_label = None
                            ts_match = re.search(
                                r'(?:received\s+at\s*)?(\d{1,2}:\d{2}\s*[AP]M)[,\s]+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})',
                                normalized_text,
                                re.IGNORECASE,
                            )
                            if ts_match:
                                time_text = ts_match.group(1).replace(' ', '')
                                date_text = ts_match.group(2).strip()
                                for fmt in ('%I:%M%p %d %b %Y', '%I:%M%p %d %B %Y'):
                                    try:
                                        parsed_dt = _dt.strptime(f'{time_text} {date_text}', fmt)
                                        parsed_txn_hour = parsed_dt.hour
                                        parsed_txn_day = parsed_dt.weekday()
                                        parsed_txn_label = parsed_dt.strftime('%I:%M %p').lstrip('0')
                                        break
                                    except ValueError:
                                        continue
                            if parsed_amount or upi_matches or parsed_txn_hour is not None:
                                parsed_bits = []
                                if parsed_amount is not None:
                                    parsed_bits.append(f"<span class='ocr-result ocr-result--amount'>Amount: ₹{parsed_amount:,.2f}</span>")
                                if parsed_sender_upi:
                                    parsed_bits.append(f"<span class='ocr-result ocr-result--upi'>Sender: {parsed_sender_upi}</span>")
                                if parsed_receiver_upi and parsed_receiver_upi != parsed_sender_upi:
                                    parsed_bits.append(f"<span class='ocr-result ocr-result--upi'>Receiver: {parsed_receiver_upi}</span>")
                                if parsed_txn_label is not None:
                                    time_label = parsed_txn_label
                                    if parsed_txn_day is not None:
                                        time_label += f" / day {parsed_txn_day}"
                                    parsed_bits.append(f"<span class='ocr-result ocr-result--amount'>Time: {time_label}</span>")
                                st.markdown(
                                    "<div class='ocr-status ocr-status--success'>Parsed OCR values:<div class='ocr-results'>" + "".join(parsed_bits) + "</div></div>",
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    "<div class='ocr-status ocr-status--info'>Auto-scoring this receipt with the trained fraud models now. The verdict will appear below once the run completes.</div>",
                                    unsafe_allow_html=True,
                                )
                                if parsed_amount is not None: st.session_state['ocr_amount'] = parsed_amount
                                if parsed_sender_upi: st.session_state['ocr_sender_upi'] = parsed_sender_upi
                                if parsed_receiver_upi: st.session_state['ocr_receiver_upi'] = parsed_receiver_upi
                                if parsed_txn_hour is not None: st.session_state['ocr_hour'] = parsed_txn_hour
                                if parsed_txn_day is not None: st.session_state['ocr_day'] = parsed_txn_day
                                # Cache OCR summary HTML for display on subsequent renders
                                st.session_state['_ocr_summary_html'] = "<div class='ocr-status ocr-status--success'>Parsed OCR values:<div class='ocr-results'>" + "".join(parsed_bits) + "</div></div>"
                                st.session_state['ocr_autoscore'] = True
                                st.session_state['ocr_autoscore_notice'] = True
                                st.rerun()
                            else:
                                st.markdown("<div class='ocr-status ocr-status--warning'>Could not extract amount or UPI ID from the screenshot.</div>", unsafe_allow_html=True)
                except ImportError:
                    st.markdown("<div class='ocr-status ocr-status--info'>pytesseract / pillow not installed.</div>", unsafe_allow_html=True)
                except Exception as _e:
                    if type(_e).__name__ == "TesseractNotFoundError":
                        st.markdown("<div class='ocr-status ocr-status--info'>Tesseract is installed in Python, but the executable was not found. Set TESSERACT_CMD or add Tesseract to PATH, then try again.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='ocr-status ocr-status--warning'>OCR error: {_e}</div>", unsafe_allow_html=True)

    with st.expander("🧾 Generate payment receipt sample"):
        rc1, rc2 = st.columns(2)
        with rc1:
            r_sender  = st.text_input("Sender UPI", "customer_sender@okicici", key="r_sender")
            r_amount  = st.number_input("Amount (₹)", 1.0, 10_000_000.0, 5000.0, 100.0, key="r_amount")
            r_app     = st.selectbox("UPI App", UPI_APPS, key="r_app")
        with rc2:
            r_receiver = st.text_input("Receiver UPI", "merchant@ybl", key="r_receiver")
            r_note     = st.text_input("Note", "Payment", key="r_note")
        if st.button("Generate Receipt", key="gen_receipt"):
            import uuid, datetime
            txn_id  = f"TXN{uuid.uuid4().hex[:12].upper()}"
            now_str = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
            st.markdown(f"""
<div style="max-width:340px;margin:auto;background:#1a1a2e;border-radius:20px;padding:28px 24px;font-family:'Inter',sans-serif;border:1px solid #2a2a4a;color:white;">
  <div style="text-align:center;margin-bottom:20px;">
    <div style="font-size:2.5rem">{'💚' if r_app=='GPay' else '💜' if r_app=='PhonePe' else '💙'}</div>
    <div style="font-weight:700;font-size:1.1rem;color:#aaa">{r_app}</div>
    <div style="font-size:2rem;font-weight:700;color:#00e676;margin:12px 0">₹{r_amount:,.0f}</div>
    <div style="background:#00e676;color:#000;border-radius:20px;padding:4px 16px;display:inline-block;font-size:0.8rem;font-weight:600">✓ Payment Successful</div>
  </div>
  <hr style="border-color:#2a2a4a;margin:16px 0"/>
  <table style="width:100%;font-size:0.85rem;color:#bbb;border-collapse:collapse;">
    <tr><td style="padding:6px 0;color:#666">From</td><td style="text-align:right;color:white">{r_sender}</td></tr>
    <tr><td style="padding:6px 0;color:#666">To</td><td style="text-align:right;color:white">{r_receiver}</td></tr>
    <tr><td style="padding:6px 0;color:#666">Date</td><td style="text-align:right;color:white">{now_str}</td></tr>
    <tr><td style="padding:6px 0;color:#666">Txn ID</td><td style="text-align:right;font-family:monospace;font-size:0.75rem;color:#888">{txn_id}</td></tr>
  </table>
  <hr style="border-color:#2a2a4a;margin:16px 0"/>
  <div style="text-align:center;font-size:0.7rem;color:#444">SAMPLE RECEIPT · FOR ANALYSIS WORKFLOW</div>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Quick-fill a scenario")
    qc1, qc2, qc3, qc4 = st.columns(4)
    SCENARIOS = {
        '🌙 Late Night Fraud':      {'sender_upi':'user9182@okicici','amount':85000.0,'hour':2,'upi_app':'GPay','merchant_category':'p2p','sender_bank':'ICICI','receiver_bank':'SBI','sender_state':'Delhi','receiver_state':'Maharashtra','velocity_override':6,'new_device':True},
        '⚡ High Velocity':          {'sender_upi':'user4471@ybl','amount':12000.0,'hour':14,'upi_app':'PhonePe','merchant_category':'p2p','sender_bank':'HDFC','receiver_bank':'Axis','sender_state':'Karnataka','receiver_state':'Karnataka','velocity_override':9,'new_device':False},
        '🗺️ Cross-State New Device': {'sender_upi':'user3310@okhdfcbank','amount':49000.0,'hour':23,'upi_app':'Paytm','merchant_category':'p2p','sender_bank':'HDFC','receiver_bank':'PNB','sender_state':'Gujarat','receiver_state':'Tamil Nadu','velocity_override':1,'new_device':True},
        '✅ Normal Transaction':     {'sender_upi':'user7723@oksbi','amount':450.0,'hour':11,'upi_app':'GPay','merchant_category':'food','sender_bank':'SBI','receiver_bank':'SBI','sender_state':'Maharashtra','receiver_state':'Maharashtra','velocity_override':1,'new_device':False},
    }
    for col, (label, _) in zip([qc1,qc2,qc3,qc4], SCENARIOS.items()):
        if col.button(label, use_container_width=True):
            st.session_state['live_scenario'] = label

    active = st.session_state.get('live_scenario')
    preset = dict(SCENARIOS.get(active, {}))
    if 'ocr_amount' in st.session_state:        preset['amount']           = st.session_state.pop('ocr_amount')
    if 'ocr_sender_upi' in st.session_state:    preset['sender_upi']       = st.session_state.pop('ocr_sender_upi')
    if 'ocr_receiver_upi' in st.session_state:  preset['receiver_upi_hint'] = st.session_state.pop('ocr_receiver_upi')
    if 'ocr_hour' in st.session_state:          preset['hour']             = st.session_state.pop('ocr_hour')
    if 'ocr_day' in st.session_state:           preset['day']              = st.session_state.pop('ocr_day')

    st.divider()

    with st.expander("📷 Parse a UPI deep link / QR payload"):
        raw_link = st.text_input("Paste UPI link", placeholder="upi://pay?pa=receiver@ybl&am=85000&tn=transfer")
        if raw_link:
            try:
                parsed = _urlparse.urlparse(raw_link)
                params = dict(_urlparse.parse_qsl(parsed.query))
                parsed_amt = None
                if 'am' in params:
                    try: parsed_amt = float(params['am'])
                    except: parsed_amt = None
                    if parsed_amt is not None:
                        if parsed_amt <= 0: st.error("Invalid amount."); parsed_amt = None
                        elif parsed_amt > UPI_MAX_LIMIT: st.error(f"Exceeds UPI max ₹{UPI_MAX_LIMIT:,}."); parsed_amt = None
                st.success(f"Parsed → receiver: `{params.get('pa','')}` | amount: ₹{params.get('am','?')}")
                if parsed_amt is not None:
                    preset = dict(preset); preset['amount'] = parsed_amt; preset['receiver_upi_hint'] = params.get('pa','')
            except Exception as _e:
                st.warning(f"Could not parse: {_e}")

    st.markdown("#### Transaction details")
    col_l, col_r = st.columns(2)
    with col_l:
        sender_upi   = st.text_input("Sender UPI ID", value=preset.get('sender_upi','user1234@okicici'))
        amount       = st.number_input("Amount (₹)", min_value=1.0, max_value=float(UPI_MAX_LIMIT), value=float(preset.get('amount',5000.0)), step=100.0)
        if amount > UPI_MERCHANT_LIMIT: st.warning(f"⚠️ ₹{amount:,.0f} exceeds UPI merchant limit.")
        upi_app      = st.selectbox("UPI App", UPI_APPS, index=UPI_APPS.index(preset.get('upi_app','GPay')) if preset.get('upi_app') in UPI_APPS else 0)
        merchant_cat = st.selectbox("Merchant Category", MERCHANT_CATS, index=MERCHANT_CATS.index(preset.get('merchant_category','p2p')) if preset.get('merchant_category') in MERCHANT_CATS else 0)
    with col_r:
        sender_bank   = st.selectbox("Sender Bank",   BANKS, index=BANKS.index(preset.get('sender_bank','SBI'))   if preset.get('sender_bank')   in BANKS else 0)
        receiver_bank = st.selectbox("Receiver Bank", BANKS, index=BANKS.index(preset.get('receiver_bank','HDFC')) if preset.get('receiver_bank') in BANKS else 1)
        sender_state   = st.selectbox("Sender State",   STATES, index=STATES.index(preset.get('sender_state','Delhi'))   if preset.get('sender_state')   in STATES else 2)
        receiver_state = st.selectbox("Receiver State", STATES, index=STATES.index(preset.get('receiver_state','Delhi'))  if preset.get('receiver_state')  in STATES else 2)
    col_t1, col_t2 = st.columns(2)
    with col_t1: txn_hour = st.slider("Transaction Hour (0–23)", 0, 23, int(preset.get('hour', _dt.now().hour)))
    with col_t2: txn_day  = st.slider("Day of Week (0=Mon)", 0, 6, int(preset.get('day', _dt.now().weekday())))

    with st.expander("⚙️ Advanced signal overrides"):
        adv_velocity    = st.slider("velocity_1h override (0=auto)", 0, 15, int(preset.get('velocity_override',0)))
        adv_unique_recv = st.slider("unique_receivers_1h override (0=auto)", 0, 15, 0)
        adv_new_device  = st.checkbox("Force is_new_device = 1", value=bool(preset.get('new_device',False)))

    st.divider()
    if st.session_state.pop('ocr_autoscore_notice', False):
        st.markdown("<div class='ocr-status ocr-status--info'>Receipt upload detected. Running the trained fraud models automatically.</div>", unsafe_allow_html=True)
    score_btn = st.button("🔍 Score this transaction", type="primary", use_container_width=True)
    _ocr_autoscore = st.session_state.get('ocr_autoscore', False)
    score_btn = score_btn or _ocr_autoscore

    # Persist verdict across reruns so it survives OCR expander state changes
    if 'live_verdict_html' in st.session_state:
        st.markdown(st.session_state['live_verdict_html'], unsafe_allow_html=True)

    if score_btn:
        st.session_state.pop('ocr_autoscore', None)
        now_str = f"2024-01-15 {txn_hour:02d}:30:00"
        row_dict = {'sender_upi':sender_upi,'receiver_upi':preset.get('receiver_upi_hint','receiver@ybl'),'amount':amount,'timestamp':now_str,'upi_app':upi_app,'merchant_category':merchant_cat,'sender_bank':sender_bank,'receiver_bank':receiver_bank,'sender_state':sender_state,'receiver_state':receiver_state}
        row_df = pd.DataFrame([row_dict])

        if _ocr_autoscore:
            st.info("OCR parsed values are being scored by the trained fraud models now: Isolation Forest, Autoencoder, LOF, and Ensemble.")

        with st.spinner("Encoding features & running all models..."):
            try:
                feature_df = encode_raw_df(row_df)
                _has_overrides = (adv_velocity > 0 or adv_unique_recv > 0 or adv_new_device)
                _SC = ['amount','amount_log','amount_vs_sender_avg','velocity_1h','velocity_24h','txn_hour','txn_day','unique_receivers_1h','amount_entropy_1h','receiver_txn_count_24h','receiver_amount_sum_24h']
                if _has_overrides:
                    feat_scaler = load_feature_scaler()
                    if feat_scaler is not None: feature_df[_SC] = feat_scaler.inverse_transform(feature_df[_SC])
                    if adv_velocity > 0:    feature_df['velocity_1h'] = adv_velocity; feature_df['velocity_24h'] = adv_velocity*3
                    if adv_unique_recv > 0: feature_df['unique_receivers_1h'] = adv_unique_recv
                    if adv_new_device:      feature_df['is_new_device'] = 1
                    if feat_scaler is not None: feature_df[_SC] = feat_scaler.transform(feature_df[_SC])

                try:
                    from src.models import run_inference
                except ImportError:
                    from models import run_inference

                _model_thresholds = {'Isolation Forest':IF_THRESHOLD,'Autoencoder':AE_THRESHOLD,'LOF':LOF_THRESHOLD,'Ensemble':THRESHOLD}
                all_results = {}
                for mn in ['Isolation Forest','Autoencoder','LOF','Ensemble']:
                    try:
                        r = run_inference(feature_df, model_name=mn, threshold=_model_thresholds[mn])
                        all_results[mn] = float(r['anomaly_score'].iloc[0])
                    except Exception as _me:
                        all_results[mn] = None
                        log.warning(f"Model {mn} failed: {_me}")

                ensemble_score   = all_results.get('Ensemble')
                # Fallback: if Ensemble failed, use IF score for verdict
                _effective_score = ensemble_score if ensemble_score is not None else all_results.get('Isolation Forest')
                is_fraud_verdict = _effective_score is not None and _effective_score >= (THRESHOLD if ensemble_score is not None else IF_THRESHOLD)
                _score_display = f"{_effective_score:.3f}" if _effective_score is not None else "Unavailable"
                _failed_models = [mn for mn, score in all_results.items() if score is None]

                # ── Verdict ──
                st.markdown("---")
                _score_label = 'Ensemble' if ensemble_score is not None else 'Isolation Forest'
                if is_fraud_verdict:
                    _verdict_html = f"<div class='verdict-fraud'><h2 style='font-size:1.8rem;margin:0 0 8px 0'>🚨 FRAUD DETECTED</h2><p style='font-size:1.1rem;margin:0'>{_score_label} score: <strong style='font-family:Courier New,monospace'>{_score_display}</strong></p></div>"
                else:
                    _verdict_html = f"<div class='verdict-normal'><h2 style='font-size:1.8rem;margin:0 0 8px 0'>✅ TRANSACTION NORMAL</h2><p style='font-size:1.1rem;margin:0'>{_score_label} score: <strong style='font-family:Courier New,monospace'>{_score_display}</strong></p></div>"
                st.session_state['live_verdict_html'] = _verdict_html
                st.markdown(_verdict_html, unsafe_allow_html=True)

                if _failed_models:
                    st.warning("Models unavailable: " + ", ".join(_failed_models))

                log_audit(page='Live Score', model='Ensemble', threshold=THRESHOLD, input_data=row_dict, n_rows=1, n_flagged=1 if is_fraud_verdict else 0, top_score=float(_effective_score) if _effective_score else 0.0, verdict='FRAUD' if is_fraud_verdict else 'NORMAL', details=json.dumps({k: float(v) if isinstance(v,(np.floating,float)) else v for k,v in all_results.items() if v is not None}))

                raw_feature_row = feature_df.iloc[0]
                tags = get_fraud_tags(raw_feature_row)
                _eff = _effective_score or 0
                _risk = "CRITICAL" if _eff>=0.85 else "HIGH" if _eff>=THRESHOLD else "MEDIUM" if _eff>=THRESHOLD*0.75 else "LOW"
                _rcol = {'CRITICAL':'#E84040','HIGH':'#FF8C00','MEDIUM':'#4A8090','LOW':'#00C896'}[_risk] if _theme=='bloomberg' else {'CRITICAL':'#CC0000','HIGH':'#8A5A00','MEDIUM':'#444','LOW':'#0B6E4F'}[_risk] if _theme=='broadsheet' else {'CRITICAL':'#E8320A','HIGH':'#E8320A','MEDIUM':'#666','LOW':'#111'}[_risk]
                _gcbg  = '#07090D' if _theme=='bloomberg' else ('#F4EFE4' if _theme=='broadsheet' else '#FFFFFF')
                _gcfg  = '#C0C8D0' if _theme=='bloomberg' else '#1A1A1A'
                _gcbar_fraud = '#E84040' if _theme=='bloomberg' else ('#CC0000' if _theme=='broadsheet' else '#E8320A')
                _gcbar_ok    = '#00C896' if _theme=='bloomberg' else ('#0B6E4F' if _theme=='broadsheet' else '#111')

                _gc1, _gc2 = st.columns([1, 1])
                with _gc1:
                    if _effective_score is not None:
                        fig_g = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=round(_effective_score, 3),
                            delta={'reference': THRESHOLD, 'valueformat':'.3f', 'increasing':{'color':'#E84040'}, 'decreasing':{'color':'#00C896'}},
                            title={'text': f"{'🚨 FRAUD DETECTED' if is_fraud_verdict else '✅ TRANSACTION NORMAL'}", 'font':{'color': _gcbar_fraud if is_fraud_verdict else _gcbar_ok, 'size':13}},
                            gauge={
                                'axis':{'range':[0,1],'tickcolor':_gcfg,'tickfont':{'color':_gcfg,'size':10}},
                                'bar':{'color': _gcbar_fraud if is_fraud_verdict else _gcbar_ok, 'thickness':0.25},
                                'bgcolor':'rgba(0,0,0,0)', 'bordercolor': _gcfg,
                                'steps':[
                                    {'range':[0, THRESHOLD*0.75], 'color':'rgba(0,200,150,0.08)'},
                                    {'range':[THRESHOLD*0.75, THRESHOLD], 'color':'rgba(255,140,0,0.08)'},
                                    {'range':[THRESHOLD, 1.0], 'color':'rgba(232,64,64,0.12)'},
                                ],
                                'threshold':{'line':{'color':_gcfg,'width':2},'thickness':0.8,'value':THRESHOLD}
                            },
                            number={'valueformat':'.3f','font':{'color':_gcfg,'family':'Courier New','size':28}}
                        ))
                        fig_g.update_layout(height=280, paper_bgcolor=_gcbg, font=dict(color=_gcfg), margin=dict(t=50,b=10,l=20,r=20))
                        st.plotly_chart(fig_g, use_container_width=True)
                    _badge_bg = {'CRITICAL':'rgba(232,64,64,0.15)','HIGH':'rgba(255,140,0,0.12)','MEDIUM':'rgba(74,96,112,0.15)','LOW':'rgba(0,200,150,0.12)'}[_risk]
                    st.markdown(f"<div style='background:{_badge_bg};border:1px solid {_rcol};border-radius:6px;padding:14px 18px;text-align:center;margin-top:4px;'><div style='color:{_rcol};font-family:Courier New,monospace;font-size:11px;letter-spacing:.2em;text-transform:uppercase;margin-bottom:6px;'>Risk Level</div><div style='color:{_rcol};font-size:2rem;font-weight:700;font-family:Courier New,monospace;'>{_risk}</div><div style='color:{_rcol};font-size:11px;margin-top:4px;opacity:.8;'>Threshold: {THRESHOLD:.2f} · Score: {_score_display} ({_score_label})</div></div>", unsafe_allow_html=True)

                with _gc2:
                    st.markdown(f"<div style='font-family:Courier New,monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px;color:{_gcfg};'>Model Scores</div>", unsafe_allow_html=True)
                    _mc_thrs = {'Isolation Forest':IF_THRESHOLD,'Autoencoder':AE_THRESHOLD,'LOF':LOF_THRESHOLD,'Ensemble':THRESHOLD}
                    _model_icons = {'Isolation Forest':'🌲','Autoencoder':'🔁','LOF':'📍','Ensemble':'⚡'}
                    for mn in ['Ensemble','Isolation Forest','Autoencoder','LOF']:
                        sc = all_results.get(mn)
                        if sc is None: continue
                        _flagged = sc >= _mc_thrs[mn]
                        _bar_pct = int(sc * 100)
                        _bar_col = _gcbar_fraud if _flagged else _gcbar_ok
                        _bg = 'rgba(232,64,64,0.08)' if _flagged else ('rgba(255,255,255,0.03)' if _theme=='bloomberg' else 'rgba(0,0,0,0.03)')
                        _bord = _gcbar_fraud if _flagged else ('rgba(255,255,255,0.07)' if _theme=='bloomberg' else 'rgba(0,0,0,0.1)')
                        st.markdown(f"<div style='background:{_bg};border:1px solid {_bord};border-radius:6px;padding:10px 14px;margin-bottom:8px;'><div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'><span style='font-family:Courier New,monospace;font-size:12px;color:{_gcfg};'>{_model_icons[mn]} {mn}</span><span style='font-family:Courier New,monospace;font-size:14px;font-weight:700;color:{_gcbar_fraud if _flagged else _gcbar_ok};'>{sc:.3f}</span></div><div style='background:rgba(128,128,128,0.15);border-radius:3px;height:5px;'><div style='background:{_bar_col};width:{_bar_pct}%;height:5px;border-radius:3px;'></div></div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-family:Courier New,monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;margin:14px 0 8px 0;color:{_gcfg};'>Triggered Signals</div>", unsafe_allow_html=True)
                    st.markdown(''.join([f'<span class="fraud-tag">{t}</span>' for t in tags]) or f"<span style='color:#666;font-size:12px;'>No specific signals</span>", unsafe_allow_html=True)

                st.markdown("---")
                checks = [
                    ('Amount',               'amount',               lambda v: f"₹{v:,.0f}",              lambda v: False),
                    ('Amount vs sender avg', 'amount_vs_sender_avg', lambda v: f"{v:.2f}×",               lambda v: v > 3),
                    ('Velocity (1h)',         'velocity_1h',          lambda v: f"{int(abs(v))} txns",     lambda v: abs(v) >= 4),
                    ('Hour',                 'txn_hour',              lambda v: f"{int(abs(v)%24)}:00",    lambda v: abs(v%24) <= 4 or abs(v%24) >= 23),
                    ('Night window',         'is_night',              lambda v: "Yes" if v>0.5 else "No",  lambda v: v > 0.5),
                    ('New device',           'is_new_device',         lambda v: "Yes" if v>0.5 else "No",  lambda v: v > 0.5),
                    ('Cross state',          'cross_state',           lambda v: "Yes" if v>0.5 else "No",  lambda v: v > 0.5),
                    ('Cross bank',           'cross_bank',            lambda v: "Yes" if v>0.5 else "No",  lambda v: v > 0.5),
                    ('Round amount',         'is_round_amount',       lambda v: "Yes" if v>0.5 else "No",  lambda v: v > 0.5),
                ]
                feat_rows = []
                for label, col_name, fmt, is_sus in checks:
                    if col_name in raw_feature_row.index:
                        disp_v = row_df.iloc[0][col_name] if col_name in row_df.columns else raw_feature_row[col_name]
                        sus = is_sus(raw_feature_row[col_name])
                        feat_rows.append({'Signal': label, 'Value': fmt(disp_v), 'Status': '🔴 Suspicious' if sus else '🟢 Normal'})
                if feat_rows:
                    _fc1, _fc2 = st.columns([1, 2])
                    with _fc1:
                        st.markdown(f"<div style='font-family:Courier New,monospace;font-size:11px;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px;color:{_gcfg};'>Why this score?</div>", unsafe_allow_html=True)
                        _explains = [f"Score {'exceeds' if is_fraud_verdict else 'is below'} threshold {THRESHOLD:.2f}"]
                        if amount > 50000: _explains.append(f"High-value ₹{amount:,.0f}")
                        if txn_hour <= 4 or txn_hour >= 23: _explains.append(f"Unusual hour {txn_hour:02d}:00")
                        if sender_state != receiver_state: _explains.append(f"Cross-state: {sender_state} → {receiver_state}")
                        for ex in _explains:
                            st.markdown(f"<div style='font-size:12px;padding:3px 0;color:{_gcfg};opacity:.85;'>→ {ex}</div>", unsafe_allow_html=True)
                    with _fc2:
                        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True, column_config={'Status': st.column_config.TextColumn(width='small')})

                # ── Session history ──
                if 'score_history' not in st.session_state: st.session_state['score_history'] = []
                st.session_state['score_history'].insert(0, {
                    'Sender': sender_upi, 'Amount (₹)': f'₹{amount:,.0f}',
                    'Hour': f'{txn_hour:02d}:00', 'App': upi_app,
                    'Score': round(ensemble_score,3) if ensemble_score else None,
                    'Verdict': '🟥 FRAUD' if is_fraud_verdict else '🟩 NORMAL',
                    'Risk': _risk,
                })
                st.session_state['score_history'] = st.session_state['score_history'][:20]

            except FileNotFoundError as e:
                st.error(f"Model not found: {e} — run `python src/models.py` first.")
            except Exception as e:
                st.error(f"Scoring error: {e}")
                import traceback; st.code(traceback.format_exc())

    # ── Session history (always visible) ──
    if st.session_state.get('score_history'):
        st.divider()
        st.markdown("#### 🕒 Session Scoring History")
        _h1, _h2 = st.columns([3,1])
        with _h1:
            st.dataframe(pd.DataFrame(st.session_state['score_history']), use_container_width=True, hide_index=True, height=min(400, 50+35*len(st.session_state['score_history'])))
        with _h2:
            _h = st.session_state['score_history']
            st.metric("Scored", len(_h))
            st.metric("🟥 Fraud",  sum(1 for r in _h if 'FRAUD' in r['Verdict']))
            st.metric("🟩 Normal", sum(1 for r in _h if 'NORMAL' in r['Verdict']))
            if st.button("Clear history", use_container_width=True):
                st.session_state['score_history'] = []
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
elif page == 'ℹ️ About':
    st.markdown("""
    <div class='hero-banner' style='padding:28px 40px'>
        <h1 style='font-size:1.8rem !important'>About PayGuard</h1>
        <p>Unsupervised fraud detection for India's UPI payment network</p>
        <span class='hero-badge'>RESEARCH PROJECT · SYNTHETIC DATA · 3 UNSUPERVISED MODELS</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### The Problem")
    c1,c2,c3 = st.columns(3)
    c1.markdown("<div class='stat-card'><h2>₹2,145Cr</h2><p>UPI fraud reported FY2024</p></div>", unsafe_allow_html=True)
    c2.markdown("<div class='stat-card'><h2>0.17%</h2><p>Fraud rate — rare but costly</p></div>", unsafe_allow_html=True)
    c3.markdown("<div class='stat-card'><h2>Days</h2><p>Delay before fraud labels arrive</p></div>", unsafe_allow_html=True)
    st.markdown("""
**Why unsupervised?** Fraud labels arrive days later via chargebacks — incomplete, delayed, adversarially noisy.
Unsupervised models learn *normal behaviour* — anything outside gets flagged, with zero dependence on historical labels.
    """)
    st.divider()

    st.markdown("### How It Works")
    steps = st.columns(7)
    for col, (icon, title, desc) in zip(steps, [('📥','Raw UPI CSV','sender, amount, app, state'),('⚙️','Feature Eng.','velocity, device, ratios'),('🔀','Split','70/15/15'),('🤖','Train','IF + AE + LOF'),('📊','Score','anomaly score'),('🚨','Flag','score ≥ threshold'),('💡','Explain','SHAP + tags')]):
        with col:
            st.markdown(f"<div class='pipeline-step'><div style='font-size:1.5rem'>{icon}</div><div style='font-weight:600;font-size:0.78rem;margin:4px 0'>{title}</div><div style='font-size:0.7rem;color:#6B7280'>{desc}</div></div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Models")
    m1,m2,m3 = st.columns(3)
    for col, icon, name, auc, desc in zip([m1,m2,m3],['🌲','🧠','🔵'],['Isolation Forest','Autoencoder','LOF'],['0.9413','0.9329','0.8624'],['Tree-based outlier detection','Neural reconstruction error','Local density anomaly scoring']):
        with col:
            st.markdown(f"<div class='stat-card'><div style='font-size:2rem'>{icon}</div><div style='font-weight:700;margin:8px 0 4px'>{name}</div><div style='font-family:monospace;font-size:1.2rem'>ROC-AUC {auc}</div><div style='font-size:0.8rem;margin:4px 0'>{desc}</div></div>", unsafe_allow_html=True)
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Dataset")
        st.markdown("""
**PayGuard — 293,000 Synthetic UPI Transactions**

No public labeled UPI fraud dataset exists. Generated using documented NPCI fraud patterns.

**6 patterns:** 🌙 Late-night P2P · ⚡ High velocity · 📱 New device · 🗺️ Cross-state · 🔄 Round amounts · 💤 Dormant spike
        """)
    with col_r:
        st.markdown("### Tech Stack")
        st.markdown("""
| Layer | Tool |
|---|---|
| ML | scikit-learn |
| Deep Learning | PyTorch |
| Explainability | SHAP |
| Visualization | Plotly |
| App | Streamlit |
        """)

    st.divider()

    # ── Feature Engineering reference ─────────────────────────────────────
    st.markdown("### Feature Engineering (22 signals)")
    feat_cols = st.columns(3)
    feature_groups = {
        "💰 Amount Signals": [
            "`amount` — raw INR value",
            "`amount_log` — log1p transform",
            "`amount_vs_sender_avg` — deviation ratio",
            "`is_round_amount` — multiples of ₹1,000",
        ],
        "⏱️ Temporal Signals": [
            "`txn_hour` — 0–23",
            "`txn_day` — 0=Mon … 6=Sun",
            "`is_weekend` — Sat/Sun flag",
            "`is_night` — 23:00–04:59",
        ],
        "⚡ Velocity Signals": [
            "`velocity_1h` — txns in last 1 h",
            "`velocity_24h` — txns in last 24 h",
            "`unique_receivers_1h` — distinct payees",
            "`amount_entropy_1h` — amount spread",
        ],
        "📱 Device & Network": [
            "`is_new_device` — unseen device ID",
            "`cross_bank` — sender ≠ receiver bank",
            "`cross_state` — sender ≠ receiver state",
            "`upi_app_enc` — GPay / PhonePe / …",
        ],
        "🏦 Receiver Signals": [
            "`receiver_txn_count_24h` — payee activity",
            "`receiver_amount_sum_24h` — payee volume",
            "`merchant_cat_enc` — category encoding",
            "`sender_bank_enc` / `receiver_bank_enc`",
        ],
        "🗺️ Geography": [
            "`sender_state_enc` — 15 states",
            "`receiver_state_enc` — 15 states",
        ],
    }
    for i, (group, feats) in enumerate(feature_groups.items()):
        with feat_cols[i % 3]:
            st.markdown(f"**{group}**")
            for f in feats:
                st.markdown(f"- {f}")

    st.divider()

    # ── Limitations ────────────────────────────────────────────────────────


    st.divider()

    # ── Audit log viewer ───────────────────────────────────────────────────
    st.markdown("### 🗄️ Audit Log")
    st.caption("Every scoring action is recorded to `outputs/audit_log.db` for traceability.")
    if os.path.exists(_audit_db_path):
        try:
            with sqlite3.connect(_audit_db_path, timeout=5) as _ac:
                _audit_df = pd.read_sql_query(
                    "SELECT timestamp, page, model, threshold, n_rows, n_flagged, top_score, verdict "
                    "FROM audit_log ORDER BY id DESC LIMIT 100",
                    _ac
                )
            if _audit_df.empty:
                st.info("No audit entries yet — score a transaction or run a batch scan.")
            else:
                _a1, _a2, _a3 = st.columns(3)
                _a1.metric("Total log entries", f"{len(_audit_df):,}")
                _fraud_entries = int((_audit_df['verdict'].str.contains('FRAUD', na=False)).sum())
                _a2.metric("Fraud verdicts logged", f"{_fraud_entries:,}")
                _a3.metric("Unique pages", _audit_df['page'].nunique())
                st.dataframe(
                    _audit_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(420, 60 + 35 * len(_audit_df)),
                )
                st.download_button(
                    "⬇ Export audit log (CSV)",
                    _audit_df.to_csv(index=False),
                    file_name="payguard_audit_log.csv",
                    mime="text/csv",
                )
        except Exception as _ae:
            st.error(f"Could not read audit log: {_ae}")
    else:
        st.info("Audit log will appear here after the first scoring action.")

    st.divider()

    # ── References & credits ───────────────────────────────────────────────
    st.markdown("### References & Credits")
    ref_c1, ref_c2 = st.columns(2)
    with ref_c1:
        st.markdown("""
**Data sources & standards**
- [NPCI UPI Ecosystem Statistics](https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics)
- [RBI Annual Report on Payment Fraud](https://rbi.org.in)
- [NPCI UPI Transaction Limits](https://www.npci.org.in/what-we-do/upi/product-overview)

**ML References**
- Liu et al. (2008) — *Isolation Forest*
- Breunig et al. (2000) — *LOF*
- Hinton & Salakhutdinov (2006) — *Autoencoders*
- Lundberg & Lee (2017) — *SHAP*
        """)
    with ref_c2:
        st.markdown("""
**Libraries used**
- `scikit-learn` — IF, LOF, metrics
- `torch` — Autoencoder
- `shap` — Explainability
- `plotly` — Interactive charts
- `streamlit` — Dashboard
- `pandas` / `numpy` — Data processing
- `sqlite3` — Audit logging
- `pytesseract` — OCR (optional)

**Fraud patterns modelled**
Based on NPCI fraud advisories and RBI cybersecurity guidelines.
        """)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.divider()
    _footer_color  = '#1A2430' if _theme == 'bloomberg' else ('#C8C0A8' if _theme == 'broadsheet' else '#E0E0E0')
    _footer_text   = '#4A6070' if _theme == 'bloomberg' else '#888'
    _footer_accent = '#FF8C00' if _theme == 'bloomberg' else ('#1A1A1A' if _theme == 'broadsheet' else '#E8320A')
    st.markdown(f"""
<div style="border-top:1px solid {_footer_color};padding:20px 0 8px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <div style="font-family:'Courier New',monospace;font-size:11px;color:{_footer_text};letter-spacing:.14em;">
    <span style="color:{_footer_accent};font-weight:700;">PAYGUARD</span>
    &nbsp;·&nbsp; Unsupervised UPI Fraud Detection
    &nbsp;·&nbsp; Synthetic data only
    &nbsp;·&nbsp; Research prototype
  </div>
  <div style="font-family:'Courier New',monospace;font-size:10px;color:{_footer_text};letter-spacing:.1em;">
    IF &nbsp;|&nbsp; AE &nbsp;|&nbsp; LOF &nbsp;|&nbsp; Ensemble &nbsp;·&nbsp; 293k txns &nbsp;·&nbsp; 22 features
  </div>
</div>
""", unsafe_allow_html=True)