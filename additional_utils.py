import os
import json
import pandas as pd

from utils.global_parameters import JSON_VCLAIMS_PATH, VCLAIMS_PATH


def json_targets_to_csv(corpus_of_json_targets, output):
    df = pd.DataFrame(columns=['id', 'vclaim', 'title'])

    for json_file in os.listdir(corpus_of_json_targets):
        json_file_path = corpus_of_json_targets + '/' + json_file
        with open(json_file_path, 'r', encoding='utf-8') as j:
            v_claim = json.loads(j.read())
            id = v_claim['vclaim_id']
            vclaim = v_claim['vclaim']
            title = v_claim['title']
            df2 = pd.DataFrame([[id, vclaim, title]], columns=['id', 'vclaim', 'title'])
            df = pd.concat([df, df2], ignore_index=True)

    df.to_csv(output, index=False, header=True, sep='\t')


#json_targets_to_csv(JSON_VCLAIMS_PATH, VCLAIMS_PATH)

def get_rid_of_long_i_claim(output_file):
    df = pd.read_csv(output_file, sep='\t', names=['id','Q0','vclaim', '1', 'score', 'tag'], dtype=str)
    df = df[df.vclaim != 'vclaim-sno-the-roommates-death']
    output = output_file + "_without_roommate"
    df.to_csv(output, index=False, header=False, sep='\t')

def get_rid_of_long_i_claim(output_file):
    df = pd.read_csv(output_file, sep='\t', names=['id','Q0','vclaim', '1', 'score', 'tag'], dtype=str)
    df = df[df.vclaim != 'vclaim-sno-the-roommates-death']
    df = df[df.vclaim != 'vclaim-sno-highest-marks']
    output = output_file + "_without_long_claims"
    df.to_csv(output, index=False, header=False, sep='\t')


def extract_title_from_vclaim(claim):
    print(claim)
    tab_position = []
    for i in range(len(claim)-2):
        if claim[i] == ' ' and claim[i+1] == ' ' and claim[i+2] == ' ':
            tab_position.append(i)
    last_tab = tab_position[len(tab_position) - 1]
    title = str(claim[last_tab+3:])
    vclaim = str(claim[: last_tab])
    return vclaim, title

def read_predict_file_float_to_int(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['id', 'Q0', 'vclaim', '1', 'score', 'tag'], dtype=str)
    df['id'] = df['id'].astype(float).astype(int)
    df['vclaim'] = df['vclaim'].astype(float).astype(int)
    df.to_csv(file_path, index=False, header=False, sep='\t')


file_path = "nlpir01/2020_2_results_1/T2-EN-nlpir01-run_type.txt"
read_predict_file_float_to_int(file_path)






