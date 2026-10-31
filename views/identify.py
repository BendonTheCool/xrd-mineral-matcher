from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
import xrd_core as core

ui.header("Identify", "Which mineral is this?",
          "Upload a powder XRD pattern. The app compares it with every reference pattern in the RRUFF "
          "library and ranks the closest minerals.")

lib = ui.library()
if lib is None:
    ui.missing_library()
    st.stop()

X, grid, meta, names = lib["X"], lib["grid"], lib["meta"], lib["names"]
pre = core.Preprocessor(grid, lib["cfg"])
_, per_query = ui.metrics()
cal = core.calibrate(per_query)

# ----------------------------------------------------------------------------- input
source = st.segmented_control("Pattern source", ["Upload a file", "Try an example"], default="Upload a file",
                              label_visibility="collapsed")

query, steps, exclude, query_label, true_name = None, None, None, "Your pattern", None

if source == "Try an example":
    counts = meta[meta.kind == "measured"].groupby("name")["sample"].nunique()
    choices = sorted(counts[counts >= 2].index)
    if not choices:
        st.info("The library has no mineral with two or more measured samples to use as an example.")
        st.stop()
    default = choices.index("Calcite") if "Calcite" in choices else 0
    pick = st.selectbox("Example mineral", choices, index=default,
                        help="Uses a real RRUFF sample. That sample's own files are hidden from the search, "
                             "so the app has to recognise it from other specimens.")
    rows = meta.index[(meta.name == pick) & (meta.kind == "measured")]
    row = int(rows[0])
    query = X[row]
    exclude = (meta["sample"] == meta["sample"].iat[row]).to_numpy()
    query_label, true_name = f"Example: {pick} ({meta.rruff_id.iat[row] or 'sample'})", pick
else:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        up = st.file_uploader("Pattern file", type=["txt", "csv", "xy", "dat"],
                              help="Two columns of numbers: 2θ in degrees, then intensity. "
                                   "Header lines are ignored. Maximum 5 MB.")
    with c2:
        wl_name = st.selectbox("X-ray source", [*core.WAVELENGTHS, "Other…"], index=0)
        if wl_name == "Other…":
            wl = st.number_input("Wavelength (Å)", min_value=0.1, max_value=5.0, value=1.5418, step=0.0001, format="%.4f")
        else:
            wl = core.WAVELENGTHS[wl_name]
        remove_bg = st.toggle("Remove background", value=True,
                              help="Subtracts a smooth baseline so only the peaks are compared. "
                                   "The reference library was built with this on.")
    if up is None:
        st.caption("Your file is processed in memory and is not saved.")
        st.stop()
    try:
        x, y = core.parse_xy_text(up.getvalue())
        steps = pre.run(x, y, wl, remove_background=remove_bg)
    except core.PatternError as e:
        st.error(str(e), icon=":material/error:")
        st.stop()
    query = steps.final
    query_label = up.name

# ----------------------------------------------------------------------------- search
hits = core.top_matches(X, names, query, k=5, exclude=exclude)
if not hits:
    st.error("No reference patterns to compare with.")
    st.stop()

best_row, best_sim = hits[0]
level, tone = core.confidence_label(best_sim, cal)
m = meta.iloc[best_row]

st.divider()
left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown('<div class="eyebrow">Best match</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="card"><div class="answer-name">{escape(m["name"])}</div>'
        f'<div class="answer-sub">{escape(core.tidy_formula(m.chemistry)) or "&nbsp;"}'
        f'{" · " + escape(m.crystal_system) if m.crystal_system and m.crystal_system != "unknown" else ""}</div>'
        f'<div style="margin-top:.8rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap">'
        f'{ui.badge(level, tone)}<span class="stat">similarity {best_sim:.3f}</span></div></div>',
        unsafe_allow_html=True)
    if true_name:
        verdict = "correct" if m["name"] == true_name else f"not correct (the sample is {true_name})"
        st.caption(f"This example is a real {true_name} sample, so the top answer is {verdict}.")
with right:
    st.markdown('<div class="eyebrow">Other candidates</div>', unsafe_allow_html=True)
    for r, (i, s) in enumerate(hits[1:], start=2):
        st.markdown(f'<div class="mono">{r}. {escape(names[i])} · {s:.3f}</div>', unsafe_allow_html=True)
    st.caption(f"Confidence levels are {cal['source']}: High ≥ {cal['high']:.3f}, Medium ≥ {cal['medium']:.3f}.")

if level == "Low":
    st.warning(
        "No confident match. This often means the sample is a mixture of minerals, is poorly crystalline, "
        "is a mineral that isn't in RRUFF, or the X-ray source setting is wrong.", icon=":material/help:")

# ----------------------------------------------------------------------------- compare (signature view)
st.subheader("Compare peaks")
options = [names[i] for i, _ in hits]
chosen = st.segmented_control("Overlay", options, default=options[0], label_visibility="collapsed") or options[0]
ci, cs = next((i, s) for i, s in hits if names[i] == chosen)
peaks = core.shared_peaks(grid, query, X[ci])
st.plotly_chart(ui.mirror_plot(grid, query, X[ci], chosen, peaks, query_label=query_label),
                use_container_width=True, config={"displaylogo": False})
st.caption(ui.describe_match(query_label, chosen, cs, peaks))
if peaks:
    st.markdown("**Peaks that line up** (2θ, Cu Kα)", help="Strong peaks found in both patterns within 0.15°. "
                "They are the main reason this mineral ranks highly.")
    st.markdown(ui.peaks_html(peaks), unsafe_allow_html=True)

# ----------------------------------------------------------------------------- table
table = pd.DataFrame([{
    "Rank": r + 1, "Mineral": names[i], "Similarity": s,
    "Chemistry": core.tidy_formula(meta.chemistry.iat[i]),
    "Crystal system": meta.crystal_system.iat[i],
    "Reference": "measured" if meta.kind.iat[i] == "measured" else "calculated (DIF)",
    "RRUFF": f"https://rruff.info/{meta.rruff_id.iat[i]}" if meta.rruff_id.iat[i] else None,
} for r, (i, s) in enumerate(hits)])
st.dataframe(table, hide_index=True, use_container_width=True, column_config={
    "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0.0, max_value=1.0, format="%.3f"),
    "RRUFF": st.column_config.LinkColumn("RRUFF", display_text=r"rruff\.info/(.*)"),
})

# ----------------------------------------------------------------------------- what happened
if steps is not None:
    with st.expander("What happened to your data"):
        c = st.columns(4)
        with c[0]:
            st.markdown('<span class="step-n">1</span> **Your file**', unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(x=steps.raw_x, y=steps.raw_y, line=dict(color=ui.REF, width=1)))
            st.plotly_chart(ui.plot_layout(fig, height=160, xtitle=None), use_container_width=True, config={"displayModeBar": False})
            st.caption(f"{len(steps.raw_x):,} points, {steps.raw_x.min():.1f}–{steps.raw_x.max():.1f}°")
        with c[1]:
            st.markdown('<span class="step-n">2</span> **Converted to Cu Kα**', unsafe_allow_html=True)
            fig = go.Figure(go.Scatter(x=ui.nan_safe(steps.cu_x), y=steps.raw_y, line=dict(color=ui.REF, width=1)))
            st.plotly_chart(ui.plot_layout(fig, height=160, xtitle=None), use_container_width=True, config={"displayModeBar": False})
            st.caption("Same d-spacings, re-expressed for the library's wavelength (Bragg's law).")
        with c[2]:
            st.markdown('<span class="step-n">3</span> **Background**', unsafe_allow_html=True)
            fig = go.Figure([go.Scatter(x=grid, y=ui.nan_safe(steps.on_grid), line=dict(color=ui.REF, width=1)),
                             go.Scatter(x=grid, y=ui.nan_safe(steps.background), line=dict(color=ui.ACCENT, width=1.5))])
            fig.update_layout(showlegend=False)
            st.plotly_chart(ui.plot_layout(fig, height=160, xtitle=None), use_container_width=True, config={"displayModeBar": False})
            st.caption("Resampled to 0.05° steps. Teal line is the baseline that gets subtracted.")
        with c[3]:
            st.markdown('<span class="step-n">4</span> **Ready to compare**', unsafe_allow_html=True)
            st.plotly_chart(ui.simple_line(grid, steps.final), use_container_width=True, config={"displayModeBar": False})
            st.caption("Square-root intensities, scaled to length 1, on the 5–90° grid.")

st.info(
    "Treat the result as a suggestion to check, not a final identification. The app assumes the sample is a single "
    "crystalline mineral that is in the RRUFF library. Mixtures, amorphous material and unusual instruments can "
    "produce confident but wrong answers.", icon=":material/info:")
st.caption("Uploaded files are processed in memory and are not saved.")
