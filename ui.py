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
                             hovertemplate="%{x:.2f}°<extra>" + escape(query_label) + "</extra>"))
    fig.add_trace(go.Scatter(x=grid, y=-ref, name=f"{ref_label} (reference, mirrored)",
                             line=dict(color=REF, width=1.4), customdata=ref,
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
