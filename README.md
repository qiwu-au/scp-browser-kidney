# Single-cell proteomics browser: mouse kidney

Interactive browser for the single-cell proteomics (SCP-MS) data of the total mouse kidney (712 cells, 4,054 proteins).

- Browser: https://scp-browser-kidney.streamlit.app/
- All browsers and the methods: https://scp-browser-landing.streamlit.app/

The About page of the browser describes the data, the cell-type annotation and how the shown values are computed.

## Run locally

With Python 3.12:

```
pip install -r requirements.txt
streamlit run app.py
```

## Files

| path | content |
| --- | --- |
| `app.py` | entry file |
| `scp_browser/` | pages, figures and summaries (code shared with the DCT browser) |
| `configs/` | texts, label sets and the About page |
| `data/kidney/` | browser data of the total kidney |
| `requirements.txt` | pinned packages |
| `.streamlit/config.toml` | theme |

In each data folder, `data_dictionary.csv` describes the downloadable files and `build_info.json` lists the data sources with their checksums.
