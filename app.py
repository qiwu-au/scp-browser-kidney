
import duckdb
import polars as pl
import plotly.express as px
import streamlit as st
import yaml
from pathlib import Path
import itertools
import io
import plotly.express as px
import plotly.graph_objects as go

# ------------------ Page & Config ------------------
st.set_page_config(page_title="Single‑Cell Proteomics Browser — Kidney", layout="wide")
CFG_PATH = Path("config.yaml")

DEFAULT_ID_LABELS = {"id1":"UniProt accession","id2":"Gene symbol","id3":"Protein description"}
DEFAULT_ID_COLS   = {"id1":"id1","id2":"id2","id3":"id3"}

if CFG_PATH.exists():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
else:
    cfg = {"data_dir":"data","id_labels":DEFAULT_ID_LABELS, "id_columns": DEFAULT_ID_COLS}

DATA_DIR   = Path(cfg.get("data_dir","data"))
ID_LABELS  = cfg.get("id_labels", DEFAULT_ID_LABELS)
ID_COLS    = cfg.get("id_columns", DEFAULT_ID_COLS)

# Safe label map for format_func in selectbox (map physical col name -> display label)
LABEL_BY_COL = {
    ID_COLS.get("id1","id1"): ID_LABELS.get("id1","id1"),
    ID_COLS.get("id2","id2"): ID_LABELS.get("id2","id2"),
    ID_COLS.get("id3","id3"): ID_LABELS.get("id3","id3"),
}

prot_parquet = DATA_DIR / "protein_summary.parquet"
cell_meta_csv = DATA_DIR / "cell_metadata.csv"
umap_parquet = DATA_DIR / "umap.parquet"
cell_qc_csv = DATA_DIR / "cell_qc.csv"

# Columns to hide in UI and to drop from CSV exports
HIDE_COLS = {"leiden", "leiden_named"}

# anchor paths to this script
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = (APP_DIR / "data").resolve()

def pq(p: Path) -> str:
    """Convert Path -> POSIX string for DuckDB."""
    return p.as_posix()

def ensure_long_view(con):
    """Create/refresh the 'long_all' view from data/long_*.parquet if missing."""
    exists = con.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = 'long_all'
        """
    ).fetchone()[0]

    if exists == 0:
        long_files = sorted(DATA_DIR.glob("long_*.parquet"))
        if long_files:
            union_sql = " UNION ALL ".join(
                [f"SELECT * FROM read_parquet('{pq(f)}')" for f in long_files]
            )
            con.execute(f"CREATE VIEW long_all AS {union_sql}")

def hide_cols(df):
    """Remove HIDE_COLS from tables shown in the UI."""
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            keep = [c for c in df.columns if c not in HIDE_COLS]
            return df.select(keep)
    except Exception:
        pass
    if hasattr(df, "drop"):  # pandas fallback
        return df.drop(columns=[c for c in HIDE_COLS if c in getattr(df, "columns", [])])
    return df

def drop_cols_pl(df):
    """Remove HIDE_COLS from a Polars DataFrame (for CSV exports)."""
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            to_drop = [c for c in HIDE_COLS if c in df.columns]
            return df.drop(to_drop) if to_drop else df
    except Exception:
        pass
    return df

def figure_square(fig, height=700):
    """Force square aspect: y axis anchored to x, set height."""
    try:
        fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10))
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    except Exception:
        pass
    return fig
    

# ------------------ DuckDB ------------------
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs; SET threads TO 4;")

if not prot_parquet.exists():
    st.error("Missing protein_summary.parquet. Run the converter first.")
    st.stop()

con.execute(f"CREATE OR REPLACE VIEW protein_summary AS SELECT * FROM read_parquet('{prot_parquet.as_posix()}');")
if cell_meta_csv.exists():
    con.execute(f"CREATE OR REPLACE VIEW cell_meta AS SELECT * FROM read_csv_auto('{cell_meta_csv.as_posix()}');")
if umap_parquet.exists():
    con.execute(f"CREATE OR REPLACE VIEW umap_raw AS SELECT * FROM read_parquet('{umap_parquet.as_posix()}');")
if cell_qc_csv.exists():
    con.execute(f"CREATE OR REPLACE VIEW cell_qc AS SELECT * FROM read_csv_auto('{cell_qc_csv.as_posix()}');")

def qi(name: str) -> str:
    "Quote an identifier for DuckDB (supports spaces/specials)."
    return '"' + str(name).replace('"','""') + '"'

# Aliased identifiers for SQL (so downstream uses id1/id2/id3 consistently)
ID1_SQL = qi(ID_COLS.get("id1","id1"))
ID2_SQL = qi(ID_COLS.get("id2","id2"))
ID3_SQL = qi(ID_COLS.get("id3","id3"))

# ------------------ Sidebar ------------------
st.title("Single‑Cell Proteomics Browser — Kidney")

datasets = [r[0] for r in con.execute("SELECT DISTINCT dataset FROM protein_summary ORDER BY dataset").fetchall()]
celltypes = [r[0] for r in con.execute("SELECT DISTINCT celltype FROM protein_summary ORDER BY celltype").fetchall()]

with st.sidebar:
    st.header("Filters")
    ds = st.multiselect("Dataset(s)", datasets, default=datasets)
    ct = st.multiselect("Cell type(s)", celltypes, default=celltypes)

    # Field select uses PHYSICAL column names as values; labels are display-only
    id_field_options = [ID_COLS.get("id1","id1"), ID_COLS.get("id2","id2"), ID_COLS.get("id3","id3")]
    field = st.selectbox("Protein ID field", id_field_options, index=0,
                     format_func=lambda col: LABEL_BY_COL.get(col, col),
                     key="id_field")
    term = st.text_input("Search term (contains)", key="search_term")


    st.markdown("**Thresholds**")
    core_pct = st.slider(
        "Core detection in selected cell type (≥ Y%)",
        min_value=0, max_value=100, value=50, step=5, key="core_pct"
    )
    unique_max_pct = st.slider(
        "Max presence in other cell types (≤ X%)",
        min_value=0, max_value=100, value=0, step=5, key="unique_max_pct",
        help="Protein must be core in the chosen cell type and present in ≤ this % of the other selected cell types."
    )
    
    # One slider to control BOTH UMAPs
    umap_px = st.slider(
        "UMAP size (px)",
        min_value=400, max_value=1200, step=50, value=700, key="umap_px",
        help="Controls the height of both UMAP plots. Width stretches to the container."
    )

core_frac = core_pct / 100.0

# ------------------ Colors ------------------
ORDERED_CELLTYPES = [r[0] for r in con.execute(
    "SELECT DISTINCT celltype FROM protein_summary ORDER BY celltype"
).fetchall()]

_base = (
    px.colors.qualitative.Plotly
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Safe
    + px.colors.qualitative.Pastel
)
PALETTE = list(itertools.islice(itertools.cycle(_base), len(ORDERED_CELLTYPES)))
CELLTYPE_COLOR = {cti: PALETTE[i] for i, cti in enumerate(ORDERED_CELLTYPES)}

def where(base: str, ds_list, ct_list):
    parts, params = [], []
    if ds_list:
        parts.append("dataset IN (" + ",".join(["?"]*len(ds_list)) + ")"); params += ds_list
    if ct_list and len(ct_list) != len(celltypes):
        parts.append("celltype IN (" + ",".join(["?"]*len(ct_list)) + ")"); params += ct_list
    if parts:
        base += " WHERE " + " AND ".join(parts)
    return base, params

# ------------------ Overview ------------------
st.subheader("Overview: mean detection rate by cell type")
with st.expander("Help — Overview", expanded=False):
    st.markdown(
        "**Figure**: Bar chart of *mean detection rate by cell type*.\n\n"
        "- For each protein, its **detection rate** within a cell type = "
        "#cells with non-null intensity ÷ #cells in that type.\n"
        "- The bar shows the **mean of these detection rates** across proteins for that cell type.\n"
        "- Use sidebar filters to alter which datasets/cell types are included."
    )

q = "SELECT celltype, AVG(detection_rate) AS mean_det FROM protein_summary"
q, p = where(q, ds, ct)
q += " GROUP BY celltype ORDER BY mean_det DESC"
df = con.execute(q, p).pl()
if df.height:
    pdf = df.to_pandas()
    fig = px.bar(
        pdf, x="celltype", y="mean_det", color="celltype",
        category_orders={"celltype": ORDERED_CELLTYPES},
        color_discrete_map=CELLTYPE_COLOR,
        labels={"celltype": "Cell type", "mean_det": "Mean detection rate (fraction of cells)"},
        title="Mean detection rate per protein (averaged over proteins)"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No data for current filters.")

# ------------------ Per‑cell QC ------------------
st.subheader("Per‑cell QC")
with st.expander("Help — Per-cell QC", expanded=False):
    st.markdown(
        "**Figures**: Violin plots by cell type.\n\n"
        "- **# proteins per cell**: distribution of identifications per cell.\n"
        "- **Dynamic range (log10)** (if shown): log10(max intensity ÷ min non-zero intensity) per cell.\n\n"
        "**Table**: One row per cell (e.g., `cell_id`, `celltype`, `dataset`, "
        "`n_proteins_identified`, and, if present, `dynamic_range_log10`)."
    )

if cell_qc_csv.exists():
    q2 = "SELECT * FROM cell_qc"
    q2, p2 = where(q2, ds, ct)
    qc = con.execute(q2, p2).pl()
    if qc.height:
        qcpdf = qc.to_pandas()
        fig2 = px.violin(
            qcpdf, x="celltype", y="n_proteins_identified", color="celltype",
            box=True, points=False,
            category_orders={"celltype": ORDERED_CELLTYPES},
            color_discrete_map=CELLTYPE_COLOR,
            labels={"celltype":"Cell type", "n_proteins_identified":"# proteins per cell"},
            title="Proteins identified per cell (by cell type)"
        )
        st.plotly_chart(fig2, width="stretch")

        if "dynamic_range_log10" in qcpdf.columns:
            fig3 = px.violin(
                qcpdf, x="celltype", y="dynamic_range_log10", color="celltype",
                box=True, points=False,
                category_orders={"celltype": ORDERED_CELLTYPES},
                color_discrete_map=CELLTYPE_COLOR,
                labels={"celltype":"Cell type", "dynamic_range_log10":"Dynamic range (log10)"},
                title="Dynamic range per cell (by cell type)"
            )
            st.plotly_chart(fig3, width="stretch")
        st.dataframe(hide_cols(qc), hide_index=True, width="stretch")
else:
    st.info("No cell_qc.csv found.")

# ------------------ Proteins (summary) — ignore never-detected ------------------
st.subheader("Proteins (summary)")
with st.expander("Help — Proteins (summary)", expanded=False):
    st.markdown(
        "**Table**: Protein × cell type summary (never-detected proteins are removed).\n\n"
        "- **Detection rate**: fraction of cells in the cell type where the protein is detected (non-null intensity).\n"
        "- **Median intensity**: median of non-zero intensities among detected cells in that type.\n"
        "- **n_detected / n_cells**: #cells detected / total #cells for that type.\n"
        "- Use the sidebar to pick the **ID field** and apply **Search term**."
    )

extra = None
if term:
    # `field` is a PHYSICAL column name; must be quoted for DuckDB
    extra = (f"lower({qi(field)}) LIKE '%%' || lower(?) || '%%'", [term])

# Alias your physical ID columns and EXCLUDE n_detected = 0
q = f"""
SELECT {ID1_SQL} AS id1,
       {ID2_SQL} AS id2,
       {ID3_SQL} AS id3,
       celltype, dataset, detection_rate, median_intensity, n_detected, n_cells
FROM protein_summary
"""
base, params = where(q, ds, ct)

# ← the only new rule you asked for:
base += (" AND " if "WHERE" in base else " WHERE ") + "n_detected > 0"

# Optional text search
if extra:
    base += " AND " + extra[0]
    params = params + extra[1]

dfp = con.execute(base, params).pl().rename({
    "id1": ID_LABELS.get("id1","id1"),
    "id2": ID_LABELS.get("id2","id2"),
    "id3": ID_LABELS.get("id3","id3"),
})
st.dataframe(hide_cols(dfp), hide_index=True, width="stretch")

# --- Helper: build UMAP table joined with metadata; safe filtering on aliased columns ---
def compute_umap_join(ds, ct):
    # Detect if the cell_meta view exists
    has_meta = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = 'cell_meta' LIMIT 1"
    ).fetchone() is not None

    if has_meta:
        inner = """
        SELECT
            u.cell_id,
            u.UMAP1, u.UMAP2,
            COALESCE(m.celltype, u.celltype) AS celltype,
            COALESCE(u.dataset,  m.dataset)  AS dataset
        FROM umap_raw u
        LEFT JOIN cell_meta m USING (cell_id)
        """
    else:
        inner = """
        SELECT
            u.cell_id,
            u.UMAP1, u.UMAP2,
            u.celltype AS celltype,
            u.dataset  AS dataset
        FROM umap_raw u
        """

    # Wrap as CTE so WHERE applies to the aliased columns from the inner SELECT
    sql = f"WITH um AS ({inner}) SELECT * FROM um"
    sql, params = where(sql, ds, ct)  # this can safely refer to `dataset`/`celltype` now
    return con.execute(sql, params).pl()

# ------------------ Cells / UMAP (square) ------------------
st.subheader("Cells / UMAP")
with st.expander("Help — Cells / UMAP", expanded=False):
    st.markdown(
        "**Figure**: 2D UMAP of single cells (colored by cell type).\n\n"
        "- Hover shows `cell_id` and `dataset`.\n"
        "- Figure is forced **square** (equal axes) for visual consistency; adjust with the size slider.\n"
        "- The table lists UMAP coordinates joined with per-cell metadata."
    )

if not umap_parquet.exists():
    st.info("No UMAP file found.")
else:
    q_um = """
    WITH um AS (
        SELECT u.cell_id, u.UMAP1, u.UMAP2,
               COALESCE(m.celltype, u.celltype) AS celltype,
               COALESCE(u.dataset, m.dataset)   AS dataset
        FROM umap_raw u
        LEFT JOIN cell_meta m USING (cell_id)
    )
    SELECT * FROM um
    """
    q_um, pu = where(q_um, ds, ct)
    um = con.execute(q_um, pu).pl()
    pdf = um.to_pandas()

    if "celltype" in pdf.columns:
        fig = px.scatter(
            pdf, x="UMAP1", y="UMAP2", color="celltype",
            hover_data=["cell_id","dataset"], title="UMAP",
            category_orders={"celltype": ORDERED_CELLTYPES},
            color_discrete_map=CELLTYPE_COLOR,
        )
    else:
        fig = px.scatter(pdf, x="UMAP1", y="UMAP2", title="UMAP")

    x_min, x_max = float(pdf["UMAP1"].min()), float(pdf["UMAP1"].max())
    y_min, y_max = float(pdf["UMAP2"].min()), float(pdf["UMAP2"].max())
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    span = max(x_max - x_min, y_max - y_min) or 1.0
    xr = [cx - span / 2.0, cx + span / 2.0]
    yr = [cy - span / 2.0, cy + span / 2.0]
    # Save axis range for the overlay to reuse
    st.session_state["umap_axis_range"] = (xr[0], xr[1], yr[0], yr[1])

    fig.update_xaxes(range=xr, constrain="domain")
    fig.update_yaxes(range=yr, scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(
    width=st.session_state.get("umap_px", 700),
    height=st.session_state.get("umap_px", 700),
    margin=dict(l=10, r=10, t=40, b=10)
)

    st.plotly_chart(fig, width="stretch")

# ==== UMAP overlay by per-cell abundance (uses sidebar 'Protein ID field' + 'Search term') ====
st.markdown("### UMAP overlay by protein/feature abundance")
ensure_long_view(con)  # make sure 'long_all' view exists

# count long rows (debug)
try:
    n_long_overlay = con.execute("SELECT COUNT(*) FROM long_all").fetchone()[0]
#   st.caption(f"(debug) rows in long_all seen by overlay: {int(n_long_overlay)}")
except Exception as e:
    n_long_overlay = 0
    st.error(f"long_all not accessible: {e}")

if n_long_overlay == 0:
    st.info("Per-cell intensities are not available. Ensure `data/long_*.parquet` exists and is readable.")
else:
    # Pull current sidebar values safely
    term = (st.session_state.get("search_term") or "").strip()
    field = st.session_state.get("id_field", ID_COLS.get("id2", "id2"))

    if not term:
        st.caption("Enter a search term in the sidebar to overlay per-cell abundance on the UMAP.")
    else:
        # Resolve candidates in protein_summary using the chosen field
        q_candidates = f"""
            SELECT DISTINCT {qi(field)} AS label
            FROM protein_summary
            WHERE LOWER({qi(field)}) LIKE '%' || LOWER(?) || '%'
            LIMIT 30
        """
        cand = con.execute(q_candidates, [term]).pl()
        if cand.height == 0:
            st.warning("No matching protein/feature for the current search.")
        else:
            labels = cand["label"].to_list()
            choice = labels[0] if len(labels) == 1 else st.selectbox("Pick a match for overlay", labels, key="overlay_choice")

            # Map UI field -> an actual column in long_all (supports id1/id2/id3 or full headers)
            long_cols = con.execute("SELECT name FROM pragma_table_info('long_all')").pl()["name"].to_list()
            candidates_in_order = [field]  # try exact header name first
            alias_map = {
                ID_COLS.get("id1","id1"): "id1",
                ID_COLS.get("id2","id2"): "id2",
                ID_COLS.get("id3","id3"): "id3",
                "UniProt accession": "id1",
                "Gene symbol": "id2",
                "Protein description": "id3",
            }
            if field in alias_map:
                candidates_in_order.append(alias_map[field])
            # final fallbacks
            for c in ["id2","id1","id3","UniProt accession","Gene symbol","Protein description"]:
                if c not in candidates_in_order:
                    candidates_in_order.append(c)

            map_field = next((c for c in candidates_in_order if c in long_cols), None)
            if map_field is None:
                st.error(
                    f"Could not map selected ID field '{field}' to any column in long_all. "
                    f"Columns found: {', '.join(map(str, long_cols))}"
                )
            else:
                # Build fresh WHERE/params locally (don’t rely on outer 'pu')
                cond_sql, cond_params = where("", ds, ct)

                # Quick sanity check: do the cell_ids overlap between umap and long?
                q_check = f"""
                    WITH um AS (SELECT DISTINCT cell_id FROM umap_raw {cond_sql}),
                         lg AS (SELECT DISTINCT cell_id FROM long_all)
                    SELECT COUNT(*) FROM um JOIN lg USING(cell_id)
                """

                overlap = con.execute(q_check, cond_params).fetchone()[0]
                if overlap == 0:
                    st.error("No overlap between UMAP cell_id and long_* cell_id. "
                             "Re-run the converter so both use the same cell_id headers.")
                else:
                    # Join per-cell intensities to FILTERED UMAP (local params ensure correct order)
                    q_overlay = f"""
                        WITH expr AS (
                          SELECT cell_id, intensity
                          FROM long_all
                          WHERE {qi(map_field)} = ?
                        )
                        SELECT u.cell_id, u.UMAP1, u.UMAP2, u.celltype, u.dataset, e.intensity
                        FROM umap_raw u
                        LEFT JOIN expr e USING (cell_id)
                        {cond_sql}
                    """
                    df_ov = con.execute(q_overlay, [choice] + cond_params).pl().to_pandas()
                    st.caption(f"overlay rows={len(df_ov)}, "
                               f"detected={(~df_ov['intensity'].isna()).sum()}, "
                               f"not_detected={(df_ov['intensity'].isna()).sum()}")

                    # Plot overlay: detected cells colored by intensity, undetected in grey
                    import numpy as np
                    if df_ov.empty:
                        st.info("No rows to display after applying current filters.")
                    else:
                        missing = df_ov[df_ov["intensity"].isna()]
                        detected = df_ov[~df_ov["intensity"].isna()]

                        fig2 = go.Figure()
                        if not missing.empty:
                            fig2.add_scattergl(
                                x=missing["UMAP1"], y=missing["UMAP2"],
                                mode="markers", name="Not detected",
                                opacity=0.35,
                                marker=dict(size=5, color="#D3D3D3"),
                                text=missing["cell_id"],
                                hovertemplate="cell: %{text}<br>not detected"
                            )
                        if not detected.empty:
                            fig2.add_scattergl(
                                x=detected["UMAP1"], y=detected["UMAP2"],
                                mode="markers", name="Intensity",
                                marker=dict(size=6, color=detected["intensity"], colorscale="Viridis", showscale=True),
                                text=detected["cell_id"],
                                hovertemplate="cell: %{text}<br>intensity: %{marker.color:.3g}"
                            )
                        # Match original UMAP axes + identical margins
                        rng = st.session_state.get("umap_axis_range")
                        if rng:
                                x0, x1, y0, y1 = rng
                                fig2.update_xaxes(range=[x0, x1], constrain="domain")
                                fig2.update_yaxes(range=[y0, y1], scaleanchor="x", scaleratio=1, constrain="domain")
  
                        # Use the sidebar size strictly for the plot area
                        desired_h = int(st.session_state.get("umap_px", 700))
                        fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10))  # same as main UMAP
                        fig2 = figure_square(fig2, height=desired_h)  # keep axes square
                        st.plotly_chart(fig2, width="stretch")        # match the main UMAP

# ------------------ Core & Unique by percentage ------------------
st.subheader("Core & Unique proteins by cell type")
with st.expander("Help — Core & Unique by %", expanded=False):
    st.markdown(
        "**Tabs**:\n"
        "- **Core (≥ Y%)**: proteins detected in at least **Y%** of cells for the selected cell type.\n"
        "- **Unique by % (≤ X%)**: among those Core proteins, keep only proteins present in **≤ X%** of the "
        "other selected cell types (computed as #other cell types with presence ÷ #other selected cell types).\n\n"
        "Tune **Y** and **X** from the sidebar *Thresholds*."
    )

# Pull presence at the core threshold within current filters; alias physical columns to id1/id2/id3
q = f"SELECT {ID1_SQL} AS id1, {ID2_SQL} AS id2, {ID3_SQL} AS id3, celltype, dataset, (detection_rate >= ?) AS present FROM protein_summary"
q, where_params = where(q, ds, ct)
params = [core_frac] + where_params
dfp = con.execute(q, params).pl()

if dfp.height == 0:
    st.info("No rows available for current filters.")
else:
    present = (
        dfp.filter(pl.col("present") == True)
           .with_columns(
               pl.concat_str([
                   pl.col("id1").cast(pl.Utf8).fill_null(""),
                   pl.col("id2").cast(pl.Utf8).fill_null(""),
                   pl.col("id3").cast(pl.Utf8).fill_null(""),
               ], separator="|").alias("pid")
           )
           .filter(pl.col("pid").str.len_bytes() > 0)
    )

    if present.height == 0:
        st.info("No proteins pass the core threshold for the current filters.")
    else:
        q_cts = "SELECT DISTINCT celltype FROM protein_summary"
        q_cts, p_cts = where(q_cts, ds, ct)
        ct_all = [str(x) for x in con.execute(q_cts, p_cts).pl()["celltype"].to_list() if x is not None]

        ct_options = sorted([str(x) for x in present["celltype"].unique().to_list() if x is not None])
        ct_choice = st.selectbox("Cell type", options=ct_options, index=0)

        core_pids = [p for p in present.filter(pl.col("celltype") == ct_choice)["pid"].unique().to_list()
                     if isinstance(p, str) and p.strip() != ""]

        other_cts = [c for c in ct_all if c != ct_choice]
        denom = len(other_cts)

        if denom == 0:
            unique_by_pct_pids = list(core_pids)
        else:
            other_presence = (
                present.filter(pl.col("celltype") != ct_choice)
                       .select(["pid", "celltype"])
                       .unique()
                       .group_by("pid")
                       .agg(pl.len().alias("n_other_cts_present"))
            )
            core_df = pl.DataFrame({"pid": core_pids})
            joined = core_df.join(other_presence, on="pid", how="left").with_columns(
                pl.col("n_other_cts_present").fill_null(0)
            )
            unique_by_pct_pids = [
                pid for pid, n in zip(joined["pid"].to_list(), joined["n_other_cts_present"].to_list())
                if (n / denom * 100.0) <= float(unique_max_pct)
            ]

        def pid_to_df(pids):
            rows = [(str(s).split("|", 2) + ["", "", ""])[:3] for s in pids]
            df = pl.DataFrame({"id1":[r[0] for r in rows],
                               "id2":[r[1] for r in rows],
                               "id3":[r[2] for r in rows]})
            return df.rename({"id1": ID_LABELS.get("id1","id1"),
                              "id2": ID_LABELS.get("id2","id2"),
                              "id3": ID_LABELS.get("id3","id3")})

        tab_core, tab_unique_pct = st.tabs([
            f"Core (≥ {core_pct}%)",
            f"Unique by % (≤ {unique_max_pct}% in other CTs)"
        ])

        with tab_core:
            core_df = pid_to_df(core_pids)
            st.caption(f"Core proteins in {ct_choice} (n={core_df.height}). Core threshold = {core_pct}%.")
            st.dataframe(core_df, hide_index=True, width="stretch")

        with tab_unique_pct:
            uniq_df = pid_to_df(unique_by_pct_pids)
            if denom > 0:
                st.caption(
                    f"Proteins in {ct_choice} (≥ {core_pct}%) and present in ≤ {unique_max_pct}% "
                    f"of the other {denom} selected cell types (n={uniq_df.height})."
                )
            else:
                st.caption(f"Only one cell type selected; treated as 0% in others (n={uniq_df.height}).")
            st.dataframe(uniq_df, hide_index=True, width="stretch")

# ------------------ Downloads ------------------
st.subheader("Downloads")
with st.expander("Help — Downloads", expanded=False):
    st.markdown(
        "Files backing the browser.\n"
        "- **protein_summary.parquet**: per-protein × cell type aggregates (columnar).\n"
        "- **cell_metadata.csv**: per-cell annotations.\n"
        "- **cell_qc.csv**: per-cell QC metrics.\n"
        "- **umap.csv**: the currently filtered UMAP + metadata."
    )

# 1) protein_summary.parquet (usually has no leiden cols; keep raw)
if prot_parquet.exists():
    st.download_button("protein_summary.parquet",
                       data=prot_parquet.read_bytes(),
                       file_name=prot_parquet.name)

# 2) cell_metadata.csv
if cell_meta_csv.exists():
    meta_df = pl.read_csv(cell_meta_csv)
    meta_clean = drop_cols_pl(meta_df)
    buf = io.StringIO(); meta_clean.write_csv(buf)
    st.download_button("cell_metadata.csv",
                       data=buf.getvalue().encode("utf-8"),
                       file_name="cell_metadata.csv", mime="text/csv")

# 3) cell_qc.csv
if cell_qc_csv.exists():
    qc_df = pl.read_csv(cell_qc_csv)
    qc_clean = drop_cols_pl(qc_df)
    buf2 = io.StringIO(); qc_clean.write_csv(buf2)
    st.download_button("cell_qc.csv",
                       data=buf2.getvalue().encode("utf-8"),
                       file_name="cell_qc.csv", mime="text/csv")

# 4) UMAP
if umap_parquet.exists():
    # if you already have compute_umap_join(ds, ct) defined, use it; else inline the same query here
    um_join = compute_umap_join(ds, ct)
    um_clean = drop_cols_pl(um_join)
    buf3 = io.StringIO(); um_clean.write_csv(buf3)
    st.download_button("umap.csv",
                       data=buf3.getvalue().encode("utf-8"),
                       file_name="umap.csv", mime="text/csv")
