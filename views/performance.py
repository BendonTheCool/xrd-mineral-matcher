import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
import xrd_core as core

ui.header("Performance", "How often is it right?",
          "Every measured sample was hidden from the library in turn and identified from the other specimens "
          "(leave-one-sample-out). These are the results from the Kaggle notebook.")

m, pq = ui.metrics()
if m is None:
    st.info("metrics.json wasn't found in the data folder. Copy it from the notebook's /kaggle/working output.",
            icon=":material/monitoring:")
    st.stop()

o = m["overall"]
c = st.columns(4)
c[0].metric("Top-1 accuracy", f"{o['top1_accuracy']:.1%}", help="The first answer is the right mineral.")
c[1].metric("Top-5 accuracy", f"{o['top5_accuracy']:.1%}", help="The right mineral is among the first five.")
c[2].metric("Test patterns", f"{o['queries']:,}")
c[3].metric("Minerals tested", f"{o['minerals_queried']:,}")


def line(xs, series: dict, xtitle, ytitle="accuracy", height=320):
    fig = go.Figure()
    dashes = ["solid", "dash", "dot"]
    for j, (name, ys) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=xs, y=ys, name=name, mode="lines+markers",
                                 line=dict(color=[ui.ACCENT, ui.REF, "#5E5CE6"][j % 3], dash=dashes[j % 3], width=2),
                                 marker=dict(symbol=["circle", "square", "diamond"][j % 3], size=8)))
    ui.plot_layout(fig, height=height, xtitle=xtitle, ytitle=ytitle)
    fig.update_layout(hovermode="x")
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return fig


def bars(labels, series: dict, height=320, ytitle="accuracy"):
    fig = go.Figure()
    for j, (name, ys) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=labels, y=ys, name=name, marker_color=[ui.ACCENT, ui.REF][j % 2],
                             marker_pattern_shape=["", "/"][j % 2], text=[f"{v:.0%}" for v in ys], textposition="outside"))
    ui.plot_layout(fig, height=height, xtitle=None, ytitle=ytitle)
    fig.update_layout(barmode="group", hovermode="x")
    fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
    return fig


def keyed(d):
    """JSON turns numeric keys into strings; sort them back numerically when possible."""
    try:
        items = sorted(d.items(), key=lambda kv: float(kv[0]))
    except ValueError:
        items = list(d.items())
    return [k for k, _ in items], [v for _, v in items]


tab_acc, tab_conf, tab_rob, tab_err = st.tabs(["Accuracy", "Confidence", "Robustness", "Errors"])

with tab_acc:
    ks = [1, 3, 5, 10]
    st.markdown(f"**Top-k accuracy.** The right mineral is the first answer {o['top1_accuracy']:.0%} of the time "
                f"and in the top 10 {o['top10_accuracy']:.0%} of the time.")
    if pq is not None and "rank_true" in pq:
        ks = list(range(1, 11))
        acc = [(pq.rank_true <= k).mean() for k in ks]
    else:
        acc = [o[f"top{k}_accuracy"] for k in ks]
    st.plotly_chart(line(ks, {"top-k accuracy": acc}, "k (number of answers shown)"), use_container_width=True)

    vk, va = keyed(m.get("vote_accuracy_by_k", {}))
    if vk:
        best = vk[int(np.argmax(va))]
        st.markdown(f"**Classic k-NN vote.** Letting the k nearest patterns vote works best at k = {best} "
                    f"({max(va):.0%}). The app ranks by each mineral's single closest pattern instead.")
        st.plotly_chart(line([int(k) for k in vk], {"top-1 accuracy": va}, "k neighbours voting"), use_container_width=True)

    abl = m.get("ablation")
    if abl:
        labels = list(abl["top1"].keys())
        st.markdown("**What the library is made of.** Accuracy using only measured patterns, only calculated (DIF) "
                    "patterns, or both.")
        st.plotly_chart(bars(labels, {"top-1": [abl["top1"][l] for l in labels],
                                      "top-5": [abl["top5"][l] for l in labels]}), use_container_width=True)

with tab_conf:
    if pq is None:
        st.info("evaluation_per_query.csv wasn't found, so the confidence charts can't be drawn.")
    else:
        cal = core.calibrate(pq)
        right, wrong = pq[pq.correct.astype(bool)], pq[~pq.correct.astype(bool)]
        st.markdown(f"**Right answers score higher.** Average top similarity is {right.sim1.mean():.3f} when the "
                    f"answer is right and {wrong.sim1.mean():.3f} when it is wrong.")
        fig = go.Figure([
            go.Histogram(x=right.sim1, name="right", marker_color=ui.ACCENT, opacity=.75, nbinsx=40),
            go.Histogram(x=wrong.sim1, name="wrong", marker_color=ui.BAD, opacity=.6, nbinsx=40,
                         marker_pattern_shape="/"),
        ])
        ui.plot_layout(fig, height=320, xtitle="similarity of the top answer", ytitle="test patterns")
        fig.update_layout(barmode="overlay", hovermode="x")
        st.plotly_chart(fig, use_container_width=True)

        s = np.sort(pq.sim1.dropna().to_numpy())
        th = np.linspace(s.min(), s.max(), 80)
        acc = [pq.correct[pq.sim1 >= t].mean() if (pq.sim1 >= t).any() else np.nan for t in th]
        cov = [(pq.sim1 >= t).mean() for t in th]
        st.markdown(f"**Trading coverage for accuracy.** Only accepting answers above a similarity threshold makes "
                    f"them more reliable but answers fewer patterns. The app calls a match High confidence above "
                    f"{cal['high']:.3f} (≥90 % right) and Medium above {cal['medium']:.3f} (≥70 % right).")
        st.plotly_chart(line(th, {"accuracy of accepted answers": acc, "share of patterns answered": cov},
                             "minimum similarity accepted", ytitle=None), use_container_width=True)

with tab_rob:
    st.markdown("Real uploads are messier than RRUFF data. The notebook disturbed the raw test patterns and "
                "re-ran the whole pipeline.")
    for key, title, xt in [("robustness_shift", "**Peak shift.** Sample-height or zero-offset errors move every peak.", "2θ shift (°)"),
                           ("robustness_noise", "**Noise.** Random noise as a fraction of the tallest peak.", "noise level"),
                           ("robustness_range", "**Shorter scans.** The instrument only measured part of the range.", "measured 2θ range")]:
        r = m.get(key)
        if not r:
            continue
        xs, t1 = keyed(r["top1"])
        _, t5 = keyed(r["top5"])
        st.markdown(title)
        if key == "robustness_range":
            st.plotly_chart(bars(xs, {"top-1": t1, "top-5": t5}), use_container_width=True)
        else:
            st.plotly_chart(line([float(v) for v in xs], {"top-1": t1, "top-5": t5}, xt), use_container_width=True)

with tab_err:
    if pq is None:
        st.info("evaluation_per_query.csv wasn't found, so the error analysis can't be shown.")
    else:
        g = pq.groupby("crystal_system").agg(top1=("correct", "mean"), top5=("in_top5", "mean"), n=("correct", "size"))
        g = g[g.n >= 10].sort_values("n", ascending=False)
        st.markdown("**By crystal system** (systems with at least 10 test patterns).")
        st.plotly_chart(bars(list(g.index), {"top-1": g.top1.tolist(), "top-5": g.top5.tolist()}), use_container_width=True)

        st.markdown("**Most common mix-ups.** Minerals with similar structures produce similar patterns.")
        wrong = pq[~pq.correct.astype(bool)]
        conf = (wrong.groupby(["truth", "pred"]).size().sort_values(ascending=False).head(15)
                .rename("times").reset_index().rename(columns={"truth": "Actual mineral", "pred": "Predicted as"}))
        st.dataframe(conf, hide_index=True, use_container_width=True)

        pm = pq.groupby("truth").agg(tests=("correct", "size"), top1=("correct", "mean"), mean_similarity=("sim1", "mean"))
        pm = pm[pm.tests >= 5].sort_values("top1")
        if len(pm):
            a, b = st.columns(2)
            a.markdown("**Hardest minerals** (≥5 tests)")
            a.dataframe(pm.head(10).style.format({"top1": "{:.0%}", "mean_similarity": "{:.3f}"}), use_container_width=True)
            b.markdown("**Easiest minerals** (≥5 tests)")
            b.dataframe(pm.tail(10).iloc[::-1].style.format({"top1": "{:.0%}", "mean_similarity": "{:.3f}"}), use_container_width=True)
