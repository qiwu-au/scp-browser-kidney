
import argparse
from pathlib import Path
import polars as pl
import yaml

def read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def lower(s: str) -> str:
    return s.lower().strip()

def clean_first_token(col: pl.Expr) -> pl.Expr:
    return col.cast(pl.Utf8).str.split(";").list.get(0).str.strip_chars()

def choose_id_columns(df: pl.DataFrame, ds_cfg: dict, id_synonyms: dict) -> tuple[list[str], dict]:
    explicit = (ds_cfg.get("id_columns") or {})
    if explicit:
        cols = [explicit.get("uniprot"), explicit.get("gene_symbol"), explicit.get("protein_name")]
        for c in cols:
            if not c or c not in df.columns:
                raise ValueError(f"[{ds_cfg['name']}] id_columns must include valid uniprot, gene_symbol, protein_name")
        return cols, {"id1":"UniProt accession","id2":"Gene symbol","id3":"Protein description"}

    col_map = {lower(c): c for c in df.columns}
    found = {}
    for key, syns in id_synonyms.items():
        for s in syns:
            if lower(s) in col_map:
                found[key] = col_map[lower(s)]
                break
    if len(found) == 3:
        return [found["uniprot"], found["gene_symbol"], found["protein_name"]], {
            "id1":"UniProt accession","id2":"Gene symbol","id3":"Protein description"
        }
    first_three = df.columns[:3]
    return [first_three[0], first_three[1], first_three[2]], {
        "id1":"UniProt accession","id2":"Gene symbol","id3":"Protein description"
    }

def detection_expr(rule: str, minv: float) -> pl.Expr:
    col = pl.col("intensity")
    finite = col.is_not_null() & ~col.is_nan()
    if rule == "non_null":
        return finite
    elif rule in ("gte_min","ge_min",">=min"):
        return finite & (col >= minv)
    else:  # gt0
        return finite & (col > 0)

def process_dataset(ds_cfg: dict, out_dir: Path, id_synonyms: dict, write_long: bool):
    name = ds_cfg["name"]
    matrix_csv = Path(ds_cfg["matrix_csv"])
    cell_meta_csv = Path(ds_cfg["cell_metadata_csv"])
    umap_csv = ds_cfg.get("umap_csv")

    if not matrix_csv.exists():
        raise FileNotFoundError(f"[{name}] Matrix CSV not found: {matrix_csv}")
    if not cell_meta_csv.exists():
        raise FileNotFoundError(f"[{name}] Cell metadata CSV not found: {cell_meta_csv}")

    print(f"[{name}] Reading matrix: {matrix_csv}")
    mat = pl.read_csv(matrix_csv, infer_schema_length=20000)

    ids, _ = choose_id_columns(mat, ds_cfg, id_synonyms)
    uni_col, gene_col, pname_col = ids

    mat = mat.with_columns([
        clean_first_token(pl.col(uni_col)).alias(uni_col),
        clean_first_token(pl.col(gene_col)).alias(gene_col),
        clean_first_token(pl.col(pname_col)).alias(pname_col),
    ])

    cell_cols = [c for c in mat.columns if c not in ids]
    if not cell_cols:
        raise ValueError(f"[{name}] No cell columns detected after ID columns: {ids}")
    long = (
        mat.unpivot(on=cell_cols, index=ids, variable_name="cell_id", value_name="intensity")
           .with_columns(pl.col("intensity").cast(pl.Float64, strict=False))
    )

    meta = pl.read_csv(cell_meta_csv)
    if "cell_id" not in meta.columns or "celltype" not in meta.columns:
        raise ValueError(f"[{name}] cell_metadata_csv must include 'cell_id' and 'celltype'. Found {meta.columns}")
    meta = meta.with_columns(pl.lit(name).alias("dataset"))
    long = long.join(meta, on="cell_id", how="left")

    long = long.rename({uni_col:"id1", gene_col:"id2", pname_col:"id3"})

    # --- Detection ---
    rule = (ds_cfg.get("detection_rule") or "gt0").lower()
    minv = float(ds_cfg.get("detection_min_value") or 0.0)

    if rule == "auto_quantile":
        q = float(ds_cfg.get("detection_quantile") or 0.2)
        scope = (ds_cfg.get("detection_quantile_scope") or "cell").lower()
        finite = long.filter(pl.col("intensity").is_not_null() & ~pl.col("intensity").is_nan())
        if scope == "dataset":
            thr = finite.group_by("dataset").agg(pl.col("intensity").quantile(q).alias("thr"))
            long = long.join(thr, on="dataset", how="left")
            det = pl.col("intensity").is_not_null() & ~pl.col("intensity").is_nan() & (pl.col("intensity") >= pl.col("thr"))
        elif scope == "global":
            thr_val = finite.select(pl.col("intensity").quantile(q).alias("thr")).item()
            det = pl.col("intensity").is_not_null() & ~pl.col("intensity").is_nan() & (pl.col("intensity") >= thr_val)
        else:
            thr = finite.group_by("cell_id").agg(pl.col("intensity").quantile(q).alias("thr"))
            long = long.join(thr, on="cell_id", how="left")
            det = pl.col("intensity").is_not_null() & ~pl.col("intensity").is_nan() & (pl.col("intensity") >= pl.col("thr"))
        long = long.with_columns(det.alias("detected"))
    else:
        long = long.with_columns(detection_expr(rule, minv).alias("detected"))

    group_keys = ["id1","id2","id3","celltype","dataset"]
    agg = (
        long.group_by(group_keys)
            .agg([
                pl.col("detected").sum().alias("n_detected"),
                pl.len().alias("n_cells"),
                pl.when(pl.col("detected")).then(pl.col("intensity")).otherwise(None).median().alias("median_intensity"),
            ])
            .with_columns((pl.col("n_detected").cast(pl.Float64) / pl.col("n_cells")).alias("detection_rate"))
    )

    keep_extras = [c for c in ["leiden","leiden_named"] if c in long.columns]
    qc = (
        long.group_by(["cell_id","dataset","celltype"] + keep_extras)
            .agg([
                pl.col("detected").sum().alias("n_proteins_identified"),
                pl.col("intensity").max().alias("intensity_max"),
                pl.when(pl.col("detected")).then(pl.col("intensity")).otherwise(None).min().alias("intensity_min_detected"),
                pl.when(pl.col("detected")).then(pl.col("intensity")).otherwise(None).median().alias("intensity_median"),
            ])
            .with_columns([
                pl.when(pl.col("intensity_min_detected").is_not_null() & (pl.col("intensity_min_detected") > 0))
                  .then((pl.col("intensity_max") / pl.col("intensity_min_detected")).log() / 2.302585092994046)
                  .otherwise(None)
                  .alias("dynamic_range_log10")
            ])
    )

    if write_long:
        long_path = out_dir / f"long_{name}.parquet"
        print(f"[{name}] Writing long matrix: {long_path}")
        long.write_parquet(long_path, compression="zstd")

    umap_df = None
    if umap_csv and Path(umap_csv).exists():
        um = pl.read_csv(umap_csv)
        needed = {"cell_id","UMAP1","UMAP2"}
        if needed.issubset(set(um.columns)):
            umap_df = (
                um.join(meta.select(["cell_id","celltype"]), on="cell_id", how="left")
                  .with_columns(pl.lit(name).alias("dataset"))
            )

    diag = long.select([
        pl.len().alias("rows_long"),
        pl.col("intensity").is_null().sum().alias("n_null"),
        pl.col("intensity").is_nan().sum().alias("n_nan"),
        pl.col("detected").cast(pl.Int64).sum().alias("n_detected_true"),
    ])
    diag_path = out_dir / f"detection_sanity_{name}.csv"
    diag.write_csv(diag_path)
    print(f"[{name}] Wrote detection sanity check: {diag_path}")

    return agg, qc, umap_df, meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    out_dir = Path(cfg.get("data_dir","data"))
    ensure_dir(out_dir)

    id_synonyms = cfg.get("id_synonyms", {
        "gene_symbol": ["gene","gene symbol","gene_symbol","symbol","gene names","genes","gn"],
        "uniprot": ["uniprot","uniprot id","uniprot_id","accession","protein ids","protein id"],
        "protein_name": ["protein name","protein names","description","protein","name"],
    })
    write_long = bool(cfg.get("write_long_parquet", False))

    all_prot, all_qc, all_umap, all_meta = [], [], [], []
    for ds in cfg["datasets"]:
        agg, qc, umap_df, meta = process_dataset(ds, out_dir, id_synonyms, write_long)
        all_prot.append(agg); all_qc.append(qc); all_meta.append(meta)
        if umap_df is not None: all_umap.append(umap_df)

    prot = pl.concat(all_prot, how="diagonal_relaxed")
    qc_all = pl.concat(all_qc, how="diagonal_relaxed")
    meta_all = pl.concat(all_meta, how="diagonal_relaxed")

    prot_out = out_dir / "protein_summary.parquet"
    qc_out = out_dir / "cell_qc.csv"
    meta_out = out_dir / "cell_metadata.csv"
    print(f"[ALL] Writing protein summary: {prot_out}")
    prot.write_parquet(prot_out, compression="zstd")
    print(f"[ALL] Writing per-cell QC: {qc_out}")
    qc_all.write_csv(qc_out)
    print(f"[ALL] Writing combined cell metadata: {meta_out}")
    meta_all.write_csv(meta_out)

    if all_umap:
        umap = pl.concat(all_umap, how="diagonal_relaxed")
        umap_out = out_dir / "umap.parquet"
        print(f"[ALL] Writing combined UMAP: {umap_out}")
        umap.write_parquet(umap_out, compression="zstd")
    else:
        print("[ALL] No UMAP provided across datasets; skipping umap.parquet")
