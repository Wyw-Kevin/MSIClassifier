# Actual Python BayesPrism implementation

This is the preserved locally patched pybayesprism0.1.0 port; vendor source and its license are included. It is not the unchanged R implementation or the latest GitHub HEAD. Historical installation commit/exact dependency snapshot were not retained; source hashes and tested current dependencies identify this snapshot.

Synthetic implementation/reference checks:

```sh
python workflows/bayesprism/validate_implementation.py --output-dir output/bayesprism
```

To reconstruct the current eight target cohorts from original inputs, prepare an isolated project root containing `GSE178341.h5ad` with `layers['counts']` and `bulk_data/<cohort>/expression.csv`, and set:

```sh
export MSI_PROJECT_ROOT=/path/to/input_project
export MSI_BAYESPRISM_WORKDIR=/path/to/new_bayesprism_run
export MSI_BAYESPRISM_COHORTS=current8
python workflows/bayesprism/prepare.py
python workflows/bayesprism/launch.py
python workflows/bayesprism/postprocess.py
```

The fixed cell/sample manifests and72×17039 counts reference are included. Original h5ad/bulk expression is not. `run_cohort.py` can execute one selected cohort; launch uses the actual recorded three-cohort/24-core configuration. Runtime helpers save sample RNG states and resume completed sample tasks. Negative/non-finite inputs stop analysis; platform inverse transforms precede whole-matrix `np.rint(...).astype(int64)`. Explicit seeds are2026, with advancing RNG state allocation.

Each cohort independently updates the reference withkey=None. Main classifier abundance is `posterior_theta_f.theta`/`final_fractions.csv`, divided by its corresponding clTopLevel parent sum; MMR4 instead uses original log2 matrices. Chain parameter1000, burn500, thin2 yields250 retained states in the archived loop. Target adaptation is cohort-specific and unsupervised. Synthetic checks do not assert real-cohort or biological measurement validity.
