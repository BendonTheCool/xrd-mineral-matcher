# XRD Mineral Matcher

A Streamlit app that identifies likely minerals from a powder X-ray diffraction pattern using k-nearest
neighbors on the RRUFF library. The model and reference library come from the Kaggle notebook
`rruff_xrd_knn.ipynb`.

## Run it

1. Run the Kaggle notebook to the end and download from `/kaggle/working`:
   `xrd_reference.npz`, `xrd_reference_meta.csv`, `xrd_config.json`, `metrics.json`, `evaluation_per_query.csv`.
2. Put them in the `data/` folder.
3. Install and start:

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Navigation and page setup |
| `xrd_core.py` | Parsing, preprocessing (identical to the notebook), k-NN search, confidence calibration |
| `ui.py` | Design tokens, CSS, the mirrored comparison chart |
| `views/identify.py` | Upload a pattern and see ranked matches |
| `views/performance.py` | Accuracy, confidence, robustness and error analysis from the notebook |
| `views/library.py` | Browse the reference minerals |
| `views/how_it_works.py` | Bragg's law, the pipeline, limitations, data credits |

## Design notes

- Colors: text `#1D1D1F` (16.8:1 on white), secondary `#6E6E73` (5.1:1), accent `#0B6E78` (6.0:1).
  Status badges are solid fills with white text (green 5.1:1, amber 5.4:1, red 6.6:1), so they work in light and dark mode,
  and each also has a shape and a word, never color alone.
- The signature element is the mirrored chart: your pattern above the axis, the reference reflected below.
- Confidence levels are calibrated on the notebook's held-out results (High = at least 90 % of answers above
  that similarity were right, Medium = at least 70 %).

## Safety

- Uploads are limited to 5 MB of text, parsed as numbers only, processed in memory and never written to disk.
- The library is loaded with `np.load(allow_pickle=False)`; no pickled objects are used.
- Text from files placed into HTML is escaped.
- If you deploy publicly, put the app behind a host with request rate limiting (Streamlit has none built in).
