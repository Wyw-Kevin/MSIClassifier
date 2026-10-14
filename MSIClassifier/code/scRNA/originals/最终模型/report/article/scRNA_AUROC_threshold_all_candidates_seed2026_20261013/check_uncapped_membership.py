from pathlib import Path
import sys
import pandas as pd
source=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
m=pd.read_csv(source/'tables/subtype_threshold_membership.csv')
for cutoff,count in zip([.60,.65,.70,.75],[39,24,17,9]):
 r=m[m.screening_cutoff.eq(cutoff)]
 expected=r.reference_oriented_AUROC.gt(cutoff)
 assert len(r)==71 and expected.sum()==count
 assert r.in_model.equals(expected), f"cutoff{cutoff}: {count} passing, {r.in_model.sum()} modeled; all passing candidates must enter"
print('PASS: threshold-only counts39/24/17/9; every candidate classified independently')
