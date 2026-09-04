"""
The house look, injected once per page. Tokens and components mirror
lazyeconomist.com (index.html) and IPS Hub (site_src/static/style.css).
"""
import streamlit as st

FONTS = ("https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;"
         "0,9..144,500;1,9..144,300;1,9..144,400&family=Inter+Tight:wght@400;500;600"
         "&family=JetBrains+Mono:wght@400;500&display=swap")

CSS = """
:root {
  --bg: #fbfaf7; --bg-soft: #f4f2ec; --ink: #1a1a1a; --ink-soft: #4a4a4a;
  --ink-faint: #8a8780; --rule: #e8e4dc; --accent: #b8410e; --accent-soft: #f5e6dd;
  --serif: 'Fraunces', Georgia, serif; --sans: 'Inter Tight', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
}
/* ---- hide Streamlit chrome ---- */
header[data-testid="stHeader"], footer, #MainMenu, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], .stDeployButton { display: none !important; }
.stApp { background: var(--bg); }
.block-container { max-width: 1120px; padding: 0 32px 64px; }
html, body, .stApp, [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  font-family: var(--sans); color: var(--ink); -webkit-font-smoothing: antialiased; }
[data-testid="stMarkdownContainer"] p { font-size: 16px; line-height: 1.55; color: var(--ink-soft); }
[data-testid="stMarkdownContainer"] a { color: var(--accent); }
h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
  font-family: var(--serif); font-weight: 400; letter-spacing: -0.02em; color: var(--ink); }
[data-testid="stMarkdownContainer"] h2 { font-size: 30px; line-height: 1.15; margin: 48px 0 8px; padding: 0; }
[data-testid="stMarkdownContainer"] h3 { font-size: 22px; margin: 32px 0 6px; padding: 0; }
[data-testid="stMarkdownContainer"] h2 em, .le-h1 em, .le-lede em { font-style: italic; color: var(--accent); font-weight: 300; }
[data-testid="stMarkdownContainer"] h2 a, [data-testid="stMarkdownContainer"] h3 a { display: none; }

/* ---- top bar ---- */
.le-nav { display: flex; justify-content: space-between; align-items: center; padding: 28px 0 20px;
  border-bottom: 1px solid var(--rule); margin-bottom: 40px; }
.le-logo { font-family: var(--serif); font-weight: 500; font-size: 19px; letter-spacing: -0.01em;
  color: var(--ink) !important; text-decoration: none !important; }
.le-logo em { font-style: italic; color: var(--accent); font-weight: 400; }
.le-logo .dot { color: var(--ink-faint); }
.le-nav ul { list-style: none; display: flex; gap: 32px; margin: 0; padding: 0; }
.le-nav ul li { margin: 0; }
.le-nav ul a { color: var(--ink-soft) !important; text-decoration: none !important; font-size: 14px; font-weight: 500; font-family: var(--sans); }
.le-nav ul a:hover, .le-nav ul a.active { color: var(--ink) !important; }
.le-nav ul a.active { border-bottom: 1.5px solid var(--accent); padding-bottom: 2px; }

/* ---- hero ---- */
.le-kicker { display: inline-flex; align-items: center; gap: 10px; font-family: var(--mono); font-size: 12px;
  color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.12em; margin: 0 0 18px; }
.le-kicker::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
.le-h1 { font-family: var(--serif); font-weight: 400; font-size: clamp(38px, 5.5vw, 64px); line-height: 1.04;
  letter-spacing: -0.025em; max-width: 820px; margin: 0 0 18px; padding: 0; color: var(--ink); }
.le-lede { font-family: var(--sans); font-size: 19px !important; color: var(--ink-soft) !important; max-width: 620px; line-height: 1.5 !important; margin: 0 0 36px; }

/* ---- tiles & cards ---- */
.le-tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0 36px; }
.le-tile { background: var(--bg-soft); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; }
.le-tile .num { font-family: var(--serif); font-size: 30px; line-height: 1.1; color: var(--ink); }
.le-tile .num.rust { color: var(--accent); font-style: italic; }
.le-tile .lbl { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ink-faint); margin-top: 6px; }
.le-note { font-family: var(--mono) !important; font-size: 12px !important; color: var(--ink-faint) !important; letter-spacing: 0.02em; }
.le-error { background: var(--accent-soft); border: 1px solid var(--accent); border-radius: 8px; color: var(--accent);
  font-family: var(--mono); font-size: 13px; padding: 12px 16px; }

/* ---- tables ---- */
.le-table { width: 100%; border-collapse: collapse; font-size: 14px; margin: 6px 0 12px; }
.le-table th { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ink-faint); text-align: left; border-bottom: 1px solid var(--rule); padding: 6px 8px 6px 0; font-weight: 500; }
.le-table td { border-bottom: 1px solid var(--rule); padding: 7px 8px 7px 0; color: var(--ink); font-family: var(--sans); }
.le-table td.num { font-family: var(--mono); text-align: right; color: var(--ink-soft); }
.le-table td.rust { color: var(--accent); }

/* ---- widgets ---- */
[data-testid="stSelectbox"] label p { font-family: var(--mono) !important; font-size: 11px !important;
  letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-faint) !important; }
[data-baseweb="select"] > div { background: #fff; border-color: var(--rule); border-radius: 8px; font-family: var(--sans); }
.le-presets { display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 24px; }
.le-presets a { font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.06em; color: var(--ink-soft) !important;
  border: 1px solid var(--rule); border-radius: 999px; padding: 5px 12px; text-decoration: none !important; }
.le-presets a:hover { border-color: var(--ink); color: var(--ink) !important; background: var(--bg-soft); }

/* ---- footer ---- */
.le-footer { border-top: 1px solid var(--rule); margin-top: 72px; padding: 28px 0 0; display: flex;
  justify-content: space-between; flex-wrap: wrap; gap: 12px; }
.le-footer .mark { font-family: var(--serif); font-size: 14px; color: var(--ink-soft); }
.le-footer .mark em { color: var(--accent); font-style: italic; }
.le-footer .meta { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); text-transform: uppercase; letter-spacing: 0.1em; }
.le-footer a { color: inherit !important; text-decoration: none !important; }

@media (max-width: 900px) { .le-tiles { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 600px) { .le-nav ul { gap: 18px; } .block-container { padding: 0 18px 48px; } .le-tiles { grid-template-columns: 1fr; } }
"""


def _one_html_block(css: str) -> str:
    """Markdown ends an HTML block at a blank line, so the CSS must contain none."""
    return "\n".join(line for line in css.splitlines() if line.strip())


def inject() -> None:
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        f'<link href="{FONTS}" rel="stylesheet"><style>{_one_html_block(CSS)}</style>',
        unsafe_allow_html=True,
    )
