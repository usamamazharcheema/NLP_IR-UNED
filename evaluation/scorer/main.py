import pdb
import logging
import argparse
import os
import pandas as pd
from trectools import TrecRun, TrecQrel, TrecEval
from os.path import join, dirname, abspath

import sys

from format_checker import check_format
from utils import print_thresholded_metric, print_single_metric


# from format_checker.main import check_format
# from utils import print_single_metric, print_thresholded_metric

sys.path.append('.')
"""
Scoring of Task 2 with the metrics Average Precision, R-Precision, P@N, RR@N. 
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


MAIN_THRESHOLDS = [1, 3, 5, 10, 20, 50, 1000]

def evaluate(gold_fpath, pred_fpath, thresholds=None):
    """
    Evaluates the predicted line rankings w.r.t. a gold file.
    Metrics are: Average Precision, R-Pr, Reciprocal Rank, Precision@N
    :param gold_fpath: the original annotated gold file, where the last 4th column contains the labels.
    :param pred_fpath: a file with line_number at each line, where the list is ordered by check-worthiness.
    :param thresholds: thresholds used for Reciprocal Rank@N and Precision@N.
    If not specified - 1, 3, 5, 10, 20, 50, len(ranked_lines).
    """
    gold_labels = TrecQrel(gold_fpath)
    prediction = TrecRun(pred_fpath)
    results = TrecEval(prediction, gold_labels)

    # Calculate Metrics
    maps = [results.get_map(depth=i) for i in MAIN_THRESHOLDS]
    mrr = results.get_reciprocal_rank()
    precisions = [results.get_precision(depth=i) for i in MAIN_THRESHOLDS]

    return maps, mrr, precisions


def validate_files(pred_file, gold_file):
    if not check_format(pred_file):
        logging.error('Bad format for pred file {}. Cannot score.'.format(pred_file))
        return False

    # Checking that all the input tweets are in the prediciton file and have predicitons. 
    pred_names = ['iclaim_id', 'zero', 'vclaim_id', 'rank', 'score', 'tag']
    pred_df = pd.read_csv(pred_file, sep='\t', names=pred_names, index_col=False)
    gold_names = ['iclaim_id', 'zero', 'vclaim_id', 'relevance']
    gold_df = pd.read_csv(gold_file, sep='\t', names=gold_names, index_col=False)
    for iclaim in set(gold_df.iclaim_id):
        if iclaim not in pred_df.iclaim_id.tolist():
            logging.error('Missing iclaim {}. Cannot score.'.format(iclaim))
            return False

    return True

# "../../data/clef_2022_checkthat_2a_english/pred_qrels.tsv"
# "../../data/clef_2022_checkthat_2a_english/gold.tsv"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, default="clef_2022_checkthat_2a_english")
    parser.add_argument('--pre_processing', action='store_true')
    parser.add_argument('--use_corpus_chunk_data', action="store_true")

    args = parser.parse_args()

    line_separator = '=' * 120

    data_path_results = "nlpir01/"
    data_path_gold = "test-input/"

    pred_file = data_path_results+args.data+"/T2-EN-nlpir01-run_type.txt"
    if args.data == "default":
        pred_file = data_path_results +  "results/T2-EN-nlpir01-run_type.txt"

    #0.1663
    #0.2364 filtering out vclaim-sno-the-roommates-death
    #0.3584 filtering out vclaim-sno-the-roommates-death and vclaim-sno-highest-marks

    gold_file = data_path_gold+args.data+"_test_input/tweet-vclaim-pairs.qrels"
    print(gold_file)
    if args.data == "default":
        gold_file = data_path_gold + "2021_2a_test_input/tweet-vclaim-pairs.qrels"

    if validate_files(pred_file, gold_file):
        maps, mrr, precisions = evaluate(gold_file, pred_file)
        filename = os.path.basename(pred_file)
        logging.info('{:=^120}'.format(' RESULTS for {} '.format(filename)))
        print_single_metric('RECIPROCAL RANK:', mrr)
        print_thresholded_metric('PRECISION@N:', MAIN_THRESHOLDS, precisions)
        print_thresholded_metric('MAP@N:', MAIN_THRESHOLDS, maps)

