import json
import os
from glob import glob

import pandas as pd


def json_vclaims_to_tsv(dir, output_dir):
    if os.path.isdir(dir):
        vclaims_fp = glob(f'{dir}/*.json')
        vclaims_fp.sort()
        vclaims = {}
        for vclaim_fp in vclaims_fp:
            with open(vclaim_fp, encoding='utf-8') as f:
                vclaim = json.load(f)
            vclaims[vclaim['vclaim_id']] = vclaim
    df = pd.DataFrame.from_dict(vclaims, orient='index')
    df = df[["vclaim", "title"]]
    df.to_csv(output_dir, sep='\t')

json_vclaims_to_tsv('data/2021_2a/vclaims', 'data/2022_2a/verified_claims.docs.tsv')