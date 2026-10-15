"""Replay current17 frozen scores and independent patient-bootstrap metrics."""
from pathlib import Path
import subprocess,sys
if __name__=='__main__':
    subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'primary17/code/verify_primary17.py'),*sys.argv[1:]],check=True)
