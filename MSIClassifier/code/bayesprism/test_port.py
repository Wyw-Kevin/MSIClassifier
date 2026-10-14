import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import numpy as np,pandas as pd
from pybayesprism.gibbs import GibbsSampler
from pybayesprism import process_input

def test_one_type_conservation():
 r=GibbsSampler.sample_Z_theta_n(np.array([10,30]),pd.DataFrame([[.25,.75]]),1,np.arange(10,20,2),2026)
 assert np.allclose(r['theta_n'],1),r['theta_n']
 assert np.allclose(r['Z_n'].sum(axis=1),[10,30])

def test_symbol_categories():
 c=process_input.assign_category(['MALAT1','XIST','ACTB','RPL3'],'hs')
 assert c.loc['MALAT1','MALAT1']

def test_protein_filter_unknown_gene():
 x=pd.DataFrame([[1,2,3]],columns=['ACTB','NOT_A_GENE_XYZ','GAPDH'])
 y=process_input.select_gene_type(x,['protein_coding'])
 assert y.columns.tolist()==['ACTB','GAPDH']

if __name__=='__main__':
 for f in [test_one_type_conservation,test_symbol_categories,test_protein_filter_unknown_gene]:
  try:f();print(f.__name__,'PASS')
  except Exception as e:print(f.__name__,'FAIL',repr(e))
