# Third-party notices

The root project license applies to original MSIClassifier software/documentation according to its stated scope. Bundled third-party material retains its original terms and notices. The root license does not remove rights granted by a third party or relicense public cohort data.

## Bundled Python BayesPrism

The actual locally patched pybayesprism0.1.0 implementation is included under `workflows/bayesprism/vendor/`. Its upstream repository is https://github.com/ziluwang829/pyBayesPrism.

The supplied upstream license is **MIT** and is preserved verbatim in:

- `workflows/bayesprism/LICENSE_pybayesprism.txt`
- `workflows/bayesprism/vendor/pybayesprism-0.1.0.dist-info/LICENSE`

The packaged upstream text contains an unfilled copyright-template line; this project preserves that record rather than inventing an upstream holder/year. Metadata, actual modified-code hashes and package identity remain available. The historical installation Git commit and complete historical dependency snapshot were not retained.

## Installed dependencies

NumPy, pandas, SciPy, scikit-learn, joblib, matplotlib and other dependencies are installed through `requirements.txt`; their software licenses remain those of their respective distributions. They are not copied into this repository under the root project license.

## Cohort-derived resources

Example/evaluation features, reference summaries and spatial records derive from the public study inputs identified by their dataset/sample IDs and provenance. Their inclusion for computational reproducibility does not replace or grant additional rights over original data sources. Raw h5ad/10x data and large TOSICA/cell2location checkpoints are supplied or acquired separately.

## Model identities

Classifier artifacts retain their actual source/seed/feature definitions. Documentation changes do not retrain or relabel them. Root model metadata is under `provenance/model_identity.json`; source/resource hashes are under `provenance/source_hashes.json`.

## Frozen TOSICA and raw-reference archive

The optional `resources/raw_reference/TOSICA_source/` retains the original TOSICA model implementation and MIT notice, copyright2022 JiaweiChenGo. Its exact upstream LICENSE is inside the separate reference archive. The customized linear layer retains its original source attribution. The archive includes actual existing annotation weights, gene masks/orders, parent dictionaries, genomic position resource and current spatial signature. It does not relabel historical training seeds or recover missing training history. Full raw data are not included.
