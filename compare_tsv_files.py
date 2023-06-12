from csv_diff import load_csv, compare

diff = compare(
    load_csv(open("2022-2b-vclaims.tsv", encoding="utf-8")),
    load_csv(open("data/2022_2b/verified_claims.docs.tsv", encoding="utf-8"))
)

print(diff.keys())
print(diff["added"])
print(diff['removed'])
print(diff['changed'])
print(diff['columns_added'])
print(diff['columns_removed'])