import os
import csv
import json

def get_vclaims(path):
    vclaims = []
    with open(path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        for row in reader:
            vclaims.append({"target_id":row[0], "target":row[1], "title":row[2]})
    return vclaims

def write_json(vclaims, path):
    n = 0
    for vclaim in vclaims:
        filename = os.path.join(path, str(n) + ".json")
        with open(filename, "w") as f:
            json.dump(vclaim, f)
        print("Wrote %s" %filename)
        n += 1
    return

def get_json_vclaims(path, fields):
    vclaims = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), "r") as f:
            vclaim = json.loads(f.read())
            vclaims.append([vclaim.get(field) for field in fields])
    return vclaims

def write_tsv(vclaims, path):
    with open(path, "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f, delimiter = "\t")
        for vclaim in vclaims:
            csv_writer.writerow(vclaim)
    print("Wrote %s" %path)

if __name__=="__main__":
    #assumes the directories json/vclaims exist
    #write_json(get_vclaims("claimlinking_riet/claimlinking_clef2020-factchecking-task2/data/v3.0/verified_claims.docs.tsv"), "claimlinking_riet/claimlinking_clef2020-factchecking-task2/data/v3.0/json/vclaims")

    #2020: id, vclaim, title
    fields = "vclaim_id", "vclaim", "title"
    write_tsv(get_json_vclaims('data/2021_2a/vclaims', fields), "2021-2a-vclaims.tsv")
    write_tsv(get_json_vclaims('data/2021_2b/politifact-vclaims', fields), "2021-2b-vclaims.tsv")
    write_tsv(get_json_vclaims('data/2022_2a/vclaims', fields), "2022-2a-vclaims.tsv")
    write_tsv(get_json_vclaims('data/2022_2b/politifact-vclaims', fields), "2022-2b-vclaims.tsv")

