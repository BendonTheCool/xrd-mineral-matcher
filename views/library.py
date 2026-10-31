import plotly.graph_objects as go
import streamlit as st

import ui
import xrd_core as core

ui.header("Library", "What the app knows",
          "Every reference pattern the app compares against, from the RRUFF Project's measured powder "
          "patterns and calculated peak lists.")

lib = ui.library()
if lib is None:
    ui.missing_library()
    st.stop()

X, grid, meta = lib["X"], lib["grid"], lib["meta"]
per_min = meta.groupby("name")["sample"].nunique()

c = st.columns(4)
c[0].metric("Minerals", f"{meta.name.nunique():,}")
c[1].metric("Samples", f"{meta['sample'].nunique():,}")
c[2].metric("Measured patterns", f"{(meta.kind == 'measured').sum():,}")
c[3].metric("Calculated patterns", f"{(meta.kind == 'dif_peaks').sum():,}")

st.subheader("Look up a mineral")
names = sorted(meta.name.unique())
pick = st.selectbox("Mineral", names, index=names.index("Quartz") if "Quartz" in names else 0,
                    placeholder="Type a mineral name", label_visibility="collapsed")
rows = meta[meta.name == pick]
first = rows.iloc[0]
info = st.columns(3)
info[0].markdown(f"**Chemistry**  \n{core.tidy_formula(first.chemistry) or '—'}")
info[1].markdown(f"**Crystal system**  \n{first.crystal_system or 'unknown'}")
info[2].markdown(f"**Samples in library**  \n{per_min[pick]}")

fig = go.Figure()
for j, (i, r) in enumerate(rows.head(6).iterrows()):
    label = f"{r.rruff_id or 'sample'} · {'measured' if r.kind == 'measured' else 'calculated'}"
    offset = -0.25 * j          # stack patterns so they can be compared
    fig.add_trace(go.Scatter(x=grid, y=X[i] + offset, name=label, line=dict(width=1.1,
                  color=ui.ACCENT if r.kind == "measured" else ui.REF, dash="solid" if r.kind == "measured" else "dot")))
ui.plot_layout(fig, height=360, ytitle=None)
fig.update_yaxes(showticklabels=False)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Up to six reference patterns for {pick}, stacked. Solid teal lines are measured, dotted grey lines "
           "are calculated from the crystal structure.")

links = [f"[{rid}](https://rruff.info/{rid})" for rid in rows.rruff_id.unique() if rid]
if links:
    st.markdown("RRUFF records: " + " · ".join(links[:12]))

st.subheader("Overview")
a, b = st.columns(2)
with a:
    fig = go.Figure(go.Histogram(x=per_min.clip(upper=20), nbinsx=20, marker_color=ui.ACCENT))
    ui.plot_layout(fig, height=300, xtitle="samples per mineral (20 = 20 or more)", ytitle="minerals")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Half of the minerals have {per_min.median():.0f} sample(s) or fewer, which limits how well "
               "rare minerals can be recognised.")
with b:
    cs = meta.drop_duplicates("sample").crystal_system.replace("", "unknown").value_counts()
    fig = go.Figure(go.Bar(x=cs.index, y=cs.values, marker_color=ui.ACCENT, text=cs.values, textposition="outside"))
    ui.plot_layout(fig, height=300, xtitle=None, ytitle="samples")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Samples by crystal system.")

m = meta[meta.kind == "measured"]
if "tt_min" in m and m.tt_min.notna().any():
    fig = go.Figure([go.Histogram(x=m.tt_min, name="scan starts", marker_color=ui.ACCENT, opacity=.75),
                     go.Histogram(x=m.tt_max, name="scan ends", marker_color=ui.REF, opacity=.75, marker_pattern_shape="/")])
    ui.plot_layout(fig, height=280, xtitle="2θ (°)", ytitle="patterns")
    fig.update_layout(barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Where measured scans start and end. Anything outside a scan counts as zero intensity.")
