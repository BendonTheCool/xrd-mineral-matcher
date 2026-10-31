"""Shared look and feel. All dynamic text placed into HTML goes through html.escape."""
from __future__ import annotations

from html import escape

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import xrd_core as core

# Design tokens (contrast measured against the surface they sit on)
ACCENT = "#0B6E78"     # crystal teal: 6.0:1 on white, white text on it 6.0:1
REF = "#8E8E93"        # reference trace (graphic only, 3.3:1 on white, meets 3:1 for non-text)
OK, WARN, BAD = "#1A7F37", "#9A5B00", "#B42318"   # white text on each: 5.1 / 5.4 / 6.6 :1

CSS = """
<style>
:root{
  --font-ui: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
  --font-mono: "SF Mono", ui-monospace, Menlo, Consolas, "Roboto Mono", monospace;
  --hairline: rgba(127,127,127,.22);
  --fill: rgba(127,127,127,.07);
}
html, body, [class*="st-"], .stMarkdown, button, input, textarea, select { font-family: var(--font-ui); }
.stMarkdown p, .stMarkdown li { font-size: 1.0625rem; line-height: 1.5; }
.block-container { max-width: 1120px; padding-top: 2.2rem; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
code, .mono { font-family: var(--font-mono); }

.eyebrow { text-transform: uppercase; letter-spacing: .06em; font-size: .78rem; font-weight: 600; opacity: .72; margin-bottom: .2rem; }
.lede { font-size: 1.2rem; line-height: 1.45; opacity: .82; max-width: 46rem; margin: 0 0 1.4rem 0; }

.card { border: 1px solid var(--hairline); background: var(--fill); border-radius: 14px; padding: 1.1rem 1.25rem; margin-bottom: .9rem; }
.answer-name { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; margin: .1rem 0 .35rem 0; }
.answer-sub { opacity: .78; font-size: 1rem; }
.stat { font-family: var(--font-mono); font-size: 1.05rem; }

.badge { display: inline-flex; align-items: center; gap: .35rem; font-weight: 600; font-size: .9rem;
         color: #fff; border-radius: 999px; padding: .22rem .7rem; }
.badge.ok { background: #1A7F37; } .badge.warn { background: #9A5B00; } .badge.bad { background: #B42318; }

.peaks span { font-family: var(--font-mono); border: 1px solid var(--hairline); border-radius: 8px;
              padding: .1rem .45rem; margin: 0 .3rem .3rem 0; display: inline-block; font-size: .95rem; }
.step-n { font-family: var(--font-mono); font-size: .85rem; opacity: .7; }
.muted { opacity: .75; }

/* ---- signature: the detector-screen hero (colors fixed: it is content, like an image) ---- */
.hero { background: radial-gradient(120% 140% at 85% 50%, #12343B 0%, #0B1F24 55%); color: #FFFFFF;
        border-radius: 22px; padding: 2rem 2.2rem; display: flex; gap: 2rem; align-items: center;
        flex-wrap: wrap; margin-bottom: 1.6rem; }
.hero-text { flex: 1 1 22rem; min-width: 0; }
.hero .eyebrow { color: #8FB3B8; opacity: 1; }                 /* 7.5:1 */
.hero h1 { color: #FFFFFF; font-size: 2.6rem; line-height: 1.08; margin: .1rem 0 .7rem 0; padding: 0; }
.hero .lede { color: #C3D5D8; opacity: 1; margin: 0; }         /* 11.2:1 */
.hero-rings { flex: 0 0 auto; text-align: center; }
.hero-rings figcaption { color: #8FB3B8; font-size: .85rem; margin-top: .4rem; font-family: var(--font-mono); }
.ring { transform-origin: 50% 50%; transform-box: view-box; animation: ring-in .7s cubic-bezier(.2,.7,.2,1) both; }
@keyframes ring-in { from { transform: scale(.55); opacity: 0; } to { transform: scale(1); } }
@media (prefers-reduced-motion: reduce) { .ring { animation: none; } }
@media (max-width: 640px) { .hero { padding: 1.4rem; } .hero h1 { font-size: 2rem; } }

/* ---- quiet surfaces ---- */
[data-testid="stMetric"] { border: 1px solid var(--hairline); background: var(--fill); border-radius: 14px; padding: .9rem 1.1rem; }
[data-testid="stMetricValue"] { font-weight: 650; letter-spacing: -0.01em; }
[data-testid="stExpander"] details { border-radius: 14px; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--hairline); border-radius: 14px; padding: .4rem .2rem 0 .2rem; }
.card { box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 6px 24px rgba(0,0,0,.05); }
.side-mark { display: flex; align-items: center; gap: .55rem; font-weight: 650; font-size: 1.02rem; margin: .2rem 0 .6rem 0; }
</style>
"""

ICON = {"ok": "●", "warn": "▲", "bad": "■"}   # shape + text + color, never color alone


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, lede: str | None = None):
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title)
    if lede:
        st.markdown(f'<p class="lede">{escape(lede)}</p>', unsafe_allow_html=True)


def rings_svg(grid, v, size=230, n_rings=16) -> str:
    """Debye–Scherrer-style rings computed from a pattern: radius from 2θ, weight and brightness from peak height."""
    from scipy.signal import find_peaks

    v = np.asarray(v, float)
    c = size / 2
    rmax = c - 6
    parts = [f'<circle cx="{c}" cy="{c}" r="{rmax:.1f}" fill="none" stroke="#1E4A52" stroke-width="1"/>']
    if v.max() > 0:
        p, props = find_peaks(v, prominence=0.05 * v.max())
        p = p[np.argsort(-props["prominences"])][:n_rings]
        h = v[p] / v[p].max() if len(p) else []
        lo, hi = float(grid[0]), float(grid[-1])
        for k, (i, hh) in enumerate(sorted(zip(p, h), key=lambda t: grid[t[0]])):
            r = 10 + (rmax - 10) * (grid[i] - lo) / (hi - lo)
            parts.append(
                f'<circle class="ring" style="animation-delay:{k * 45}ms" cx="{c}" cy="{c}" r="{r:.1f}" fill="none" '
                f'stroke="#5CC8D1" stroke-opacity="{0.25 + 0.75 * hh:.2f}" stroke-width="{0.8 + 3.2 * hh:.2f}"/>')
    parts.append(f'<circle cx="{c}" cy="{c}" r="5" fill="#0B1F24" stroke="#8FB3B8" stroke-width="1.5"/>')  # beam stop
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="Diffraction rings drawn from the strongest peaks">' + "".join(parts) + "</svg>")


def hero(eyebrow: str, title: str, lede: str, svg: str, caption: str):
    st.markdown(
        f'<section class="hero"><div class="hero-text"><div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1><p class="lede">{escape(lede)}</p></div>'
        f'<figure class="hero-rings" style="margin:0">{svg}<figcaption>{escape(caption)}</figcaption></figure></section>',
        unsafe_allow_html=True)


def sidebar_mark():
    mark = ('<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">'
            '<circle cx="13" cy="13" r="11.5" fill="#0B1F24"/>'
            '<circle cx="13" cy="13" r="8.5" fill="none" stroke="#5CC8D1" stroke-width="1.6"/>'
            '<circle cx="13" cy="13" r="5" fill="none" stroke="#5CC8D1" stroke-opacity=".6" stroke-width="1"/>'
            '<circle cx="13" cy="13" r="1.8" fill="#8FB3B8"/></svg>')
    st.sidebar.markdown(f'<div class="side-mark">{mark}<span>XRD Mineral Matcher</span></div>', unsafe_allow_html=True)


def badge(label: str, tone: str) -> str:
    return f'<span class="badge {tone}" role="status"><span aria-hidden="true">{ICON[tone]}</span>{escape(label)} confidence</span>'


def peaks_html(peaks) -> str:
    return '<div class="peaks">' + "".join(f"<span>{p:.2f}°</span>" for p in peaks) + "</div>"


@st.cache_resource(show_spinner="Loading the reference library…")
def library():
    return core.load_library()


@st.cache_data(show_spinner=False)
def metrics():
    return core.load_metrics()


def missing_library():
    st.warning("The reference library isn't loaded yet, so the app can't identify anything.", icon=":material/database:")
    st.markdown(
        f"""
Run the Kaggle notebook to the end, then download these files from `/kaggle/working` and put them in
`{escape(str(core.DATA_DIR))}`:

- `xrd_reference.npz` and `xrd_reference_meta.csv` (required)
- `xrd_config.json` (preprocessing settings)
- `metrics.json` and `evaluation_per_query.csv` (for the Performance page and calibrated confidence)

Then reload this page.
"""
    )


def plot_layout(fig: go.Figure, height=380, xtitle="2θ (°, Cu Kα)", ytitle=None):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        font=dict(family="-apple-system, BlinkMacSystemFont, system-ui, Segoe UI, sans-serif", size=14),
    )
    fig.update_xaxes(title=xtitle, showgrid=False, zeroline=False)
    fig.update_yaxes(title=ytitle, showgrid=True, gridcolor="rgba(127,127,127,.15)", zeroline=False)
    return fig


def mirror_plot(grid, query, ref, ref_label, peaks=(), query_label="Your pattern"):
    """The signature view: query above the axis, reference reflected below it."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=query, name=query_label, line=dict(color=ACCENT, width=1.6),
                             fill="tozeroy", fillcolor="rgba(11,110,120,.14)",
                             hovertemplate="%{x:.2f}°<extra>" + escape(query_label) + "</extra>"))
    fig.add_trace(go.Scatter(x=grid, y=-ref, name=f"{ref_label} (reference, mirrored)",
                             line=dict(color=REF, width=1.4), customdata=ref,
                             fill="tozeroy", fillcolor="rgba(142,142,147,.14)",
                             hovertemplate="%{x:.2f}°<extra>" + escape(ref_label) + "</extra>"))
    for p in peaks:
        fig.add_vline(x=p, line=dict(color="rgba(127,127,127,.35)", width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color="rgba(127,127,127,.6)", width=1))
    plot_layout(fig, height=420, ytitle="relative intensity")
    fig.update_yaxes(showticklabels=False)
    return fig


def describe_match(query_label, ref_label, sim, peaks):
    """Text alternative for the mirror chart (charting-data.md: describe charts in words too)."""
    s = f"{query_label} compared with {ref_label}: cosine similarity {sim:.3f}."
    if peaks:
        s += " Peaks that line up: " + ", ".join(f"{p:.2f}°" for p in peaks) + "."
    return s


def simple_line(grid, y, color=ACCENT, height=160, name=""):
    fig = go.Figure(go.Scatter(x=grid, y=y, line=dict(color=color, width=1.2), name=name, showlegend=False))
    plot_layout(fig, height=height, xtitle=None)
    fig.update_yaxes(showticklabels=False)
    return fig


def nan_safe(a):
    a = np.asarray(a, float)
    return np.where(np.isfinite(a), a, None)
