import streamlit as st

import ui

ui.header("How It Works", "From peaks to a mineral name",
          "X-rays bounce off the layers of atoms in a crystal. Each mineral's layer spacings produce peaks at "
          "its own set of angles, a fingerprint the app can look up.")

st.subheader("Bragg's law")
st.latex(r"n\lambda = 2d\sin\theta")
st.markdown("A peak appears at angle 2θ when the X-ray wavelength λ fits the spacing d between atomic layers. "
            "The d-spacings belong to the mineral, while the angles also depend on the X-ray source. That is why the "
            "app converts every pattern to one wavelength (copper Kα) before comparing.")

st.subheader("The pipeline")
steps = [
    ("Read", "Your file is read as two columns: angle 2θ and intensity."),
    ("Convert", "Angles are converted to copper Kα using Bragg's law, keeping the same d-spacings."),
    ("Clean", "A smooth background is subtracted and the pattern is resampled to 0.05° steps from 5° to 90°."),
    ("Scale", "Intensities are square-rooted so small peaks still count, then the pattern is scaled to length 1."),
    ("Match", "k-nearest neighbors finds the reference patterns with the highest cosine similarity and ranks minerals."),
]
cols = st.columns(len(steps))
for n, (col, (title, text)) in enumerate(zip(cols, steps), start=1):
    col.markdown(f'<span class="step-n">{n}</span>', unsafe_allow_html=True)
    col.markdown(f"**{title}**")
    col.caption(text)

st.subheader("Why k-nearest neighbors")
st.markdown(
    "Identifying a mineral is already a comparison: find the known pattern that looks most like yours. k-NN does "
    "exactly that, and every answer points to the real reference patterns behind it, so you can check the match "
    "yourself with the mirror plot. It also needs no training, which matters because many minerals in RRUFF have "
    "only one or two samples.")

st.subheader("What it can't do")
st.markdown(
    "- **Mixtures.** It assumes one mineral. A rock with several minerals gives a blend of patterns.\n"
    "- **Minerals that aren't in the library.** It will still return the closest mineral it knows.\n"
    "- **Amorphous material** such as glass has no sharp peaks to match.\n"
    "- **Very different instruments** or strong preferred orientation change peak heights and can reduce accuracy.")

st.subheader("Data and privacy")
st.markdown(
    "Reference data comes from the [RRUFF Project](https://rruff.info) (Lafuente, Downs, Yang and Stone, 2015). "
    "Uploaded files are processed in memory to make the prediction and are not stored. The app has no accounts and "
    "collects no personal information.")
st.caption("XRD measurements use ionising radiation. Only operate diffractometers with training and supervision.")
