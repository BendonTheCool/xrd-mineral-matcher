import streamlit as st

import ui

ui.header("How It Works", "From peaks to a mineral name",
          "X-rays bounce off the layers of atoms in a crystal. Each mineral's layer spacings produce peaks at "
          "its own set of angles, a fingerprint the app can look up.")

st.subheader("Bragg's law")
st.latex(r"n\lambda = 2d\sin\theta")
_atoms = "".join(f'<circle cx="{40 + 36 * i}" cy="{y}" r="5" fill="currentColor" opacity=".55"/>'
                 for y in (80, 130) for i in range(10))
st.markdown(
    '<figure style="margin:.4rem 0 1rem 0"><svg viewBox="0 0 420 170" width="100%" style="max-width:520px" role="img" '
    'aria-label="X-rays reflecting from two layers of atoms a distance d apart, at angle theta">'
    + _atoms +
    '<line x1="40" y1="80" x2="370" y2="80" stroke="currentColor" stroke-opacity=".25"/>'
    '<line x1="40" y1="130" x2="370" y2="130" stroke="currentColor" stroke-opacity=".25"/>'
    '<polyline points="60,20 184,80 308,20" fill="none" stroke="#0B6E78" stroke-width="2.5"/>'
    '<polyline points="80,20 204,130 328,20" fill="none" stroke="#0B6E78" stroke-width="2.5" stroke-dasharray="6 4"/>'
    '<line x1="388" y1="80" x2="388" y2="130" stroke="currentColor" stroke-width="1.5"/>'
    '<text x="396" y="110" fill="currentColor" font-size="15" font-style="italic">d</text>'
    '<text x="140" y="74" fill="currentColor" font-size="15" font-style="italic">θ</text>'
    '<text x="40" y="160" fill="currentColor" font-size="12" opacity=".75">layers of atoms in the crystal</text>'
    '</svg><figcaption style="font-size:.9rem;opacity:.75">The lower ray travels an extra 2d·sinθ. When that '
    'equals a whole number of wavelengths the two rays add up and a peak appears.</figcaption></figure>',
    unsafe_allow_html=True)
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
