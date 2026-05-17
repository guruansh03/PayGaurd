"""
themes.py — Three switchable UI themes for PayGuard dashboard.
Bloomberg Terminal (default), Broadsheet, Swiss Editorial.
"""


def hex_to_rgba(hex_color, alpha=0.1):
    """Convert #RRGGBB to rgba(r,g,b,a) for Plotly compatibility."""
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = ''.join([c*2 for c in h])
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

THEMES = {
    'Bloomberg Terminal': {
        'bg': '#0E131A', 'sidebar_bg': '#07090D', 'sidebar_border': '#1A2430',
        'text': '#C0C8D0', 'text2': '#2A4050', 'accent': '#FF8C00',
        'ok': '#00C896', 'danger': '#E84040', 'card_bg': 'rgba(7,9,13,0.85)',
        'card_border': '#121C26', 'font': "'Courier New',Courier,monospace",
        'rad': '0px', 'input_bg': '#121C26', 'grid': '#121C26',
        'tab_bg': '#07090D', 'tab_active': '#121C26',
        'hero_bg': 'linear-gradient(135deg,#07090D 0%,#0E131A 100%)',
        'hero_text': '#FF8C00', 'hero_sub': '#2A4050',
        'kpi_blue_bg': 'rgba(26,48,64,0.4)', 'kpi_blue_border': '#FF8C00',
        'kpi_red_bg': 'rgba(232,64,64,0.1)', 'kpi_red_border': '#E84040',
        'kpi_green_bg': 'rgba(0,200,150,0.1)', 'kpi_green_border': '#00C896',
        'kpi_amber_bg': 'rgba(255,140,0,0.1)', 'kpi_amber_border': '#FF8C00',
        'verdict_fraud_bg': 'linear-gradient(135deg,#3D0A0A 0%,#E84040 100%)',
        'verdict_ok_bg': 'linear-gradient(135deg,#0A2E1E 0%,#00C896 100%)',
        'btn_bg': '#FF8C00', 'btn_text': '#07090D',
        'sidebar_text': '#C0C8D0', 'sidebar_accent': '#FF8C00',
    },
    'Broadsheet': {
        'bg': '#F4EFE4', 'sidebar_bg': '#1A1A1A', 'sidebar_border': '#333',
        'text': '#1A1A1A', 'text2': '#666', 'accent': '#1A1A1A',
        'ok': '#1A1A1A', 'danger': '#1A1A1A', 'card_bg': '#F4EFE4',
        'card_border': '#1A1A1A', 'font': "'Times New Roman',Times,serif",
        'rad': '0px', 'input_bg': '#EDE8DA', 'grid': '#C8C0A8',
        'tab_bg': '#EDE8DA', 'tab_active': '#F4EFE4',
        'hero_bg': '#1A1A1A',
        'hero_text': '#F4EFE4', 'hero_sub': '#888',
        'kpi_blue_bg': '#F4EFE4', 'kpi_blue_border': '#1A1A1A',
        'kpi_red_bg': '#F4EFE4', 'kpi_red_border': '#1A1A1A',
        'kpi_green_bg': '#F4EFE4', 'kpi_green_border': '#1A1A1A',
        'kpi_amber_bg': '#F4EFE4', 'kpi_amber_border': '#1A1A1A',
        'verdict_fraud_bg': '#1A1A1A',
        'verdict_ok_bg': '#1A1A1A',
        'btn_bg': '#1A1A1A', 'btn_text': '#F4EFE4',
        'sidebar_text': '#F4EFE4', 'sidebar_accent': '#F4EFE4',
    },
    'Swiss Editorial': {
        'bg': '#FFFFFF', 'sidebar_bg': '#111', 'sidebar_border': '#333',
        'text': '#111', 'text2': '#999', 'accent': '#E8320A',
        'ok': '#111', 'danger': '#E8320A', 'card_bg': '#FFFFFF',
        'card_border': '#E0E0E0', 'font': "'Helvetica Neue',Helvetica,Arial,sans-serif",
        'rad': '0px', 'input_bg': '#F5F5F5', 'grid': '#E0E0E0',
        'tab_bg': '#F5F5F5', 'tab_active': '#111',
        'hero_bg': '#111',
        'hero_text': '#FFF', 'hero_sub': '#999',
        'kpi_blue_bg': '#FFF', 'kpi_blue_border': '#E8320A',
        'kpi_red_bg': '#FFF', 'kpi_red_border': '#E8320A',
        'kpi_green_bg': '#FFF', 'kpi_green_border': '#111',
        'kpi_amber_bg': '#FFF', 'kpi_amber_border': '#999',
        'verdict_fraud_bg': '#E8320A',
        'verdict_ok_bg': '#111',
        'btn_bg': '#E8320A', 'btn_text': '#FFF',
        'sidebar_text': '#FFF', 'sidebar_accent': '#E8320A',
    },
}


def get_theme_css(name):
    """Return complete CSS string for the given theme."""
    t = THEMES[name]
    is_bb = name == 'Bloomberg Terminal'
    sidebar_label = t['sidebar_text']

    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{{font-family:{t['font']};}}
[data-testid="stMarkdown"] p,[data-testid="stMarkdown"] li,
[data-testid="stMarkdown"] h1,[data-testid="stMarkdown"] h2,
[data-testid="stMarkdown"] h3,[data-testid="stMarkdown"] h4,
[data-testid="stText"],[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] p{{font-family:{t['font']}!important;color:{t['text']}!important;}}

[data-testid="stAppViewContainer"],[data-testid="stMain"],
.main .block-container{{background:{t['bg']}!important;}}

[data-testid="stSidebar"],[data-testid="stSidebar"]>div{{
  background:{t['sidebar_bg']}!important;border-right:1px solid {t['sidebar_border']}!important;}}
[data-testid="stSidebar"] [data-testid="stMarkdown"] p,
[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
[data-testid="stSidebar"] .stRadio label span,
[data-testid="stSidebar"] .stSlider [data-testid="stMarkdownContainer"] p{{
  color:{sidebar_label}!important;}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"]>div{{
  background:{t['input_bg']}!important;color:{t['text']}!important;border-color:{t['card_border']}!important;}}

[data-testid="metric-container"]{{
  background:{t['card_bg']}!important;border:1.5px solid {t['card_border']}!important;
  border-radius:{t['rad']}!important;padding:20px!important;
  transition:transform .25s ease,box-shadow .25s ease!important;}}
[data-testid="metric-container"]:hover{{transform:translateY(-3px)!important;}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{
  font-family:'Space Mono',monospace!important;font-size:1.8rem!important;color:{t['text']}!important;}}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{{color:{t['text2']}!important;}}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p{{color:{t['text2']}!important;}}

.stTabs [data-baseweb="tab-list"]{{background:transparent!important;gap:8px;}}
.stTabs [data-baseweb="tab"]{{
  background:{t['tab_bg']}!important;border:1px solid {t['card_border']};
  border-radius:{t['rad']};color:{t['text2']}!important;transition:all .2s ease;}}
.stTabs [aria-selected="true"]{{
  background:{t['tab_active']}!important;border-color:{t['accent']}!important;
  color:{t['text']}!important;font-weight:500;}}

[data-testid="stDataFrame"],.stDataFrame{{
  border-radius:{t['rad']}!important;overflow:hidden;}}

[data-testid="stExpander"]{{border:1px solid {t['card_border']}!important;border-radius:{t['rad']}!important;}}
[data-testid="stExpander"] summary>span>[data-testid="stMarkdownContainer"] p{{color:{t['text']}!important;}}

.stAlert p{{color:{t['text']}!important;}}

.stButton>button[kind="primary"]{{
  background:{t['btn_bg']}!important;color:{t['btn_text']}!important;
  border:none!important;border-radius:{t['rad']}!important;
  font-family:{t['font']}!important;letter-spacing:.06em;text-transform:uppercase;font-weight:700;
  transition:all .2s ease!important;}}
.stButton>button[kind="primary"]:hover{{
  filter:brightness(1.15)!important;transform:translateY(-1px)!important;}}

[data-testid="stHorizontalRule"]{{border-color:{t['card_border']}!important;}}

.hero-banner{{
  background:{t['hero_bg']};border-radius:{t['rad']};
  padding:{'10px 14px' if is_bb else '28px 40px'};margin-bottom:{'0' if is_bb else '28px'};
  position:relative;overflow:hidden;
  {'border-bottom:1px solid '+t['card_border']+';' if is_bb else ''}}}
.hero-banner h1{{color:{t['hero_text']}!important;
  font-size:{'13px' if is_bb else '1.8rem'}!important;
  font-weight:700!important;letter-spacing:{'0.12em' if is_bb else 'normal'};
  margin:0!important;font-family:{t['font']}!important;}}
.hero-banner p{{color:{t['hero_sub']}!important;margin:0!important;
  font-size:{'10px' if is_bb else '1rem'};font-family:{t['font']}!important;}}
.hero-banner .hero-badge{{
  display:inline-block;background:rgba(255,255,255,0.15);
  border:1px solid rgba(255,255,255,0.25);border-radius:{'0' if is_bb else '20px'};
  padding:3px 14px;font-size:{'8px' if is_bb else '0.72rem'};
  color:{t['hero_sub']};font-weight:500;letter-spacing:0.08em;margin-top:{'4px' if is_bb else '12px'};
  text-transform:uppercase;font-family:{t['font']}!important;}}

.fraud-tag{{
  display:inline-flex;align-items:center;gap:4px;
  background:{'rgba(232,64,64,0.14)' if is_bb else 'rgba(26,26,26,0.06)' if name=='Broadsheet' else 'rgba(232,50,10,0.06)'};
  border:1px solid {t['danger']+'33'};color:{t['danger']}!important;
  border-radius:{t['rad']};padding:4px 14px;font-size:{'8px' if is_bb else '0.78rem'};
  font-weight:600;margin:3px;letter-spacing:.06em;
  font-family:{t['font']}!important;text-transform:{'uppercase' if is_bb else 'none'};}}

.stat-card{{
  background:{t['card_bg']};border:1.5px solid {t['card_border']};
  border-radius:{t['rad']};padding:24px 28px;margin-bottom:12px;
  transition:all .3s ease;}}
.stat-card:hover{{transform:translateY(-4px);box-shadow:0 8px 24px {t['accent']}15;}}
.stat-card h2{{font-family:'Space Mono',monospace!important;margin:0;font-size:2rem;color:{t['accent']}!important;
  -webkit-text-fill-color:{t['accent']}!important;}}
.stat-card p{{color:{t['text2']}!important;margin:4px 0 0;font-size:0.82rem;
  text-transform:uppercase;letter-spacing:.08em;}}

.pipeline-step{{
  background:{t['card_bg']};border:2px solid {t['card_border']};
  border-radius:{t['rad']};padding:18px 12px;text-align:center;transition:all .3s ease;}}
.pipeline-step:hover{{border-color:{t['accent']};transform:scale(1.05);}}

.kpi-card{{border-radius:{t['rad']};padding:24px;text-align:center;
  transition:all .25s ease;font-family:{t['font']}!important;}}
.kpi-card:hover{{transform:translateY(-3px);}}
.kpi-blue{{background:{t['kpi_blue_bg']};border-left:4px solid {t['kpi_blue_border']};}}
.kpi-red{{background:{t['kpi_red_bg']};border-left:4px solid {t['kpi_red_border']};}}
.kpi-green{{background:{t['kpi_green_bg']};border-left:4px solid {t['kpi_green_border']};}}
.kpi-amber{{background:{t['kpi_amber_bg']};border-left:4px solid {t['kpi_amber_border']};}}
.kpi-icon{{font-size:1.6rem;margin-bottom:6px;}}
.kpi-value{{font-family:'Space Mono',monospace;font-size:1.9rem;font-weight:700;color:{t['text']};}}
.kpi-label{{font-size:{'8px' if is_bb else '0.78rem'};color:{t['text2']};
  text-transform:uppercase;letter-spacing:{'0.14em' if is_bb else '0.06em'};margin-top:4px;}}
.kpi-delta{{font-size:0.75rem;color:{t['text2']};margin-top:2px;}}

.verdict-fraud{{
  background:{t['verdict_fraud_bg']};color:white!important;
  border-radius:{t['rad']};padding:28px 32px;text-align:center;}}
.verdict-fraud h2,.verdict-fraud p{{color:white!important;}}
.verdict-normal{{
  background:{t['verdict_ok_bg']};color:white!important;
  border-radius:{t['rad']};padding:28px 32px;text-align:center;}}
.verdict-normal h2,.verdict-normal p{{color:white!important;}}

.section-header{{font-family:{t['font']}!important;font-size:{'8px' if is_bb else '0.7rem'};
  text-transform:uppercase;letter-spacing:{'0.14em' if is_bb else '0.15em'};
  color:{t['accent'] if is_bb else t['text2']}!important;margin-bottom:8px;font-weight:500;}}

[data-testid="stSidebar"] .stRadio>div{{gap:2px;}}
[data-testid="stSidebar"] .stRadio label{{
  border-left:3px solid transparent;padding-left:12px;
  border-radius:0 {t['rad']} {t['rad']} 0;transition:all .2s;}}
[data-testid="stSidebar"] .stRadio label:has(input:checked){{
  border-left-color:{t['sidebar_accent']};
  background:linear-gradient(90deg,{t['sidebar_accent']}14 0%,transparent 100%);}}

.score-badge{{font-family:'Space Mono',monospace;font-size:0.8rem;font-weight:500;padding:2px 8px;border-radius:{t['rad']};}}
.score-high{{background:{t['danger']}1A;color:{t['danger']}!important;}}
.score-mid{{background:{t['accent']}1A;color:{t['accent']}!important;}}
.score-low{{background:{t['ok']}1A;color:{t['ok']}!important;}}

.stSpinner>div{{border-color:{t['accent']} transparent transparent transparent!important;}}

/* Theme toggle buttons */
.theme-toggle-bar{{display:flex;gap:4px;margin:8px 0 12px;}}
.theme-toggle-bar button{{
  flex:1;padding:6px 8px;font-size:9px;font-weight:700;letter-spacing:.08em;
  cursor:pointer;border:1.5px solid transparent;text-transform:uppercase;
  transition:all .18s;font-family:'Courier New',monospace;}}
</style>"""


def theme_hero_html(name, title, subtitle, badge='', extra_html=''):
    """Return theme-specific hero/header HTML."""
    t = THEMES[name]
    if name == 'Bloomberg Terminal':
        ticker = extra_html or ''
        return f"""<div class='hero-banner' style='display:flex;align-items:center;gap:10px;'>
            <span style='font-size:13px;font-weight:700;color:#FF8C00;letter-spacing:.12em;
              font-family:Courier New,monospace;'>{title}</span>
            {ticker}
            <span style='margin-left:auto;font-size:10px;color:#2A4050;
              font-family:Courier New,monospace;'>{subtitle}</span>
        </div>"""
    elif name == 'Broadsheet':
        return f"""<div class='hero-banner'>
            <div style='font-size:22px;color:#F4EFE4;font-style:italic;
              letter-spacing:-.02em;font-family:Times New Roman,serif;'>{title}</div>
            <div style='font-size:9px;color:#888;letter-spacing:.22em;text-transform:uppercase;
              font-family:Courier New,monospace;margin-top:4px;'>{subtitle}</div>
            {'<div style="font-size:9px;color:#666;letter-spacing:.08em;margin-top:4px;font-family:Courier New,monospace;">'+badge+'</div>' if badge else ''}
        </div>"""
    else:  # Swiss
        return f"""<div class='hero-banner' style='display:flex;align-items:center;gap:0;'>
            <div style='font-size:15px;font-weight:700;color:#FFF;letter-spacing:.28em;
              text-transform:uppercase;font-family:Helvetica Neue,sans-serif;'>{title}</div>
            <div style='margin-left:auto;font-size:9px;color:#999;letter-spacing:.18em;
              text-transform:uppercase;'>{subtitle}</div>
        </div>
        <div style='height:4px;background:#E8320A;'></div>"""


def chart_colors(name):
    """Return dict for Plotly chart theming."""
    t = THEMES[name]
    return {
        'paper': t['bg'], 'plot': t['bg'],
        'text': t['text'], 'grid': t['grid'],
        'font': t['font'].split(',')[0].strip("'"),
    }
