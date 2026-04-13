from __future__ import annotations
from utility_scripts import HMMLoader
import polars
import subprocess
import os
from pathlib import Path
import urllib.request
import tempfile
import argparse

def _save_hmm_file(model_path, hmms):
    # filter hmms to only include those in the model
    hmm_out_file = f"{model_path}/features"
    model_features = polars.read_csv(f"{model_path}/domains.tsv", separator='\t')
    # open and save hmms
    hmms_to_save = HMMLoader.read_hmms(hmmdb_path=Path(hmms), whitelist=model_features.to_series().to_list())
    hmms_to_save.write_to_h3m_file(hmm_out_file)

    return hmm_out_file + ".selected.h3m"


if __name__ == "__main__":
    model_path = "src/data/results/gecco/model_0"
    hmms = "src/data/hmms"
    print(_save_hmm_file(model_path, hmms))
