# SMT-GraphFormer: Spatiotemporal Multi-Task Graph Transformer for Transit Prediction

This directory is a standalone public slice of the transit M24 pipeline used for the SMTGraphFormer paper. It keeps the copied code as close as possible to the original project, keeping the same file structure and names where possible. The code is not intended to be used as a general purpose library, but rather as a reference implementation for the paper.

The project expects one external input. Place the raw stop level pickle file in `data/` with the original filename `atbData-May2024-stopLevel-[fPM.eST.eLU.eDW].pkl`. All other artefacts are built inside this repository.

The intended workflow is straightforward. First run `notebooks/dataIntegration.ipynb` to build the transformed stop data, split plan, transform bundle, trip level SMT data, and relational matrices inside `data/`. After that you can use `notebooks/trainModel.ipynb` or `scripts/trainModel.py` for SMT training, and `notebooks/bmXGB.ipynb` or `notebooks/bmRTDL.ipynb` for the paper benchmarks.

The top level layout is as follows:
- The Python package lives in `src/smtgraphformer/`. 
- Training and evaluation code (notebooks, scripts) in `notebooks/`. Training configs in `configs/`. 
- The built data artefacts are exported to `data/`, and saved models go in `models/`.
