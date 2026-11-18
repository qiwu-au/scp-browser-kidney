## About this browser

Single-cell proteomics (SCP-MS) of the total mouse kidney, as reported in the accompanying manuscript (title and DOI to be added on publication).

- Raw data: ProteomeXchange / PRIDE **PXD073729**.
- Processed objects: Zenodo (DOI to be added on publication).
- Code: [github.com/qiwu-au/scp_mouse_kidney_dct](https://github.com/qiwu-au/scp_mouse_kidney_dct).

### Data

- **Cells:** 768 cells were acquired. The 712 cells with at least 100 detected proteins are retained.
- **Proteins:** 4,054 of the 4,144 protein groups are kept. These are the ones detected in at least 2 retained cells.
- **Intensities:** Spectronaut PG.Quantity per cell, with no imputation. "Detected" means intensity > 0.
- **UMAP:**
  - Analysis matrix: ln(1 + intensity), z-scored per protein and clipped at ±10.
  - Neighbour graph: k = 10 on 20 principal components.
  - The UMAP is the stored embedding of that graph. It was computed without the cell labels.

### Cell types

- **Clustering:** Leiden 2.0 clusters on the stored graph (seed 42; minimum 5 cells).
- **Naming:** each cluster is named from its differential proteins (that cluster vs all other cells) and a curated kidney marker list (Supplementary Table S2).
- **Counts:** PCT 160, PST 158, TAL 138, DCT 58, CNT/PC 38, JG 24, Endothelial 19, IC-B 8, Unknown 109.
- **Label codes:** *DCT* and *CNT/PC* have the codes DCT1 and PC in the downloads.
- **Not resolved at cluster level with protein evidence:** DTL, ATL, MD, DCT1 vs DCT2, IC-A, DVR, AVR and podocytes.
- **Sensitive label:** the JG label is lost after removing 20–28 random cells in 58–69% of draws.
- The *Clusters and their marker evidence* table on the Overview page lists the evidence for every cluster.

### How values are computed

- **Detection rate:** the fraction of a cell type's cells with intensity > 0.
- **Median intensity:** computed over the detected cells only.
- **Mean ln(1 + intensity):** computed over all cells of the cell type, with not-detected cells counted as 0.
- **Dotplots:**
  - Size is the detection rate.
  - Colour is the cell-type mean of the per-protein z-scored ln(1 + intensity), rescaled 0–1 per protein across the plotted rows.
  - The marker presets reproduce the values of the cell-type marker dotplot (differences below 10⁻⁶).
- **Core & Unique:**
  - Core = detected in at least Y% of a cell type's cells.
  - Unique = Core in at most X% of the other selected cell types.

### Version

The data files were built from the analysis objects by the browser's data build script. The build date and the checksum of every data source are in `build_info.json` (Downloads page).
