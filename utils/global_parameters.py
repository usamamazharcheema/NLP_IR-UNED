MODEL_PATH = "saved_models/pre_trained_model"

TWEETS_TRAIN_PATH = "data/2020_2/train/tweets.queries.tsv"
GOLD_LABELS_TRAIN_PATH = "data/2020_2/train/tweet-vclaim-pairs.qrels"
TWEETS_DEV_PATH = "data/2020_2/dev/tweets.queries.tsv"
GOLD_LABELS_DEV_PATH = "data/2020_2/dev/tweet-vclaim-pairs.qrels"

#TWEETS_TEST_PATH = "2021_2a_test_input/2021_2a_test_input/tweets.queries.tsv"
#GOLD_LABELS_TEST_PATH = "2021_2a_test_input/2021_2a_test_input/tweet-vclaim-pairs.qrels"
# TWEETS_TEST_PATH = "2021_2a_test_input/2020_2_test_input/tweets.queries.tsv"
# GOLD_LABELS_TEST_PATH = "2021_2a_test_input/2020_2_test_input/tweet-vclaim-pairs.qrels"

# CHANGE THIS
TWEETS_TEST_PATH = "test-input/2022_2b_test_input/tweets.queries.tsv"
GOLD_LABELS_TEST_PATH = "test-input/2022_2b_test_input/tweet-vclaim-pairs.qrels"

#JSON_VCLAIMS_PATH = "data/v3.0/corpus"

# AND THIS
VCLAIMS_PATH = "data/2020_2/verified_claims.docs.tsv"

# JSON_VCLAIMS_PATH = "data/2020_2/corpus"
#
VCLAIMS_TEST_PATH = "data/2022_2b/verified_claims.docs.tsv"

#AND THIS
RESULTS_FILE_PATH = "nlpir01/2021_2a/nlpir01.results"
PREDICT_SUB_FILE = "nlpir01/2021_2a/T2-EN-nlpir01-run_type.txt"
#RESULTS_FILE_PATH = "nlpir01/2020_2_results/nlpir01.results"

TRAIN_DF_FILE = "nlpir01/data/train_df.pkl"
DEV_DF_FILE = "nlpir01/data/dev_df.pkl"
TEST_DF_FILE = "nlpir01/data/test_df.pkl"
TRAIN_JSON_FILE = "nlpir01/data/train.json"
DEV_JSON_FILE = "nlpir01/data/dev.json"
TEST_JSON_FILE = "nlpir01/data/test.json"

PREDICT_FILE = "nlpir01/2021_2a/nlpir01.predictions"
RESULT_FILE = "nlpir01/2021_2a/nlpir01.result"
#PREDICT_SUB_FILE = "nlpir01/2022_2a/T2-EN-nlpir01-run_type.txt"
TRAINING_PNG_FILE = "nlpir01/2021_2a/nlpir01.training_acc_loss.png"
# PREDICT_FILE = "nlpir01/2020_2_results/nlpir01.predictions"
# RESULT_FILE = "nlpir01/2020_2_results/nlpir01.result"
# PREDICT_SUB_FILE = "nlpir01/2020_2_results/T2-EN-nlpir01-run_type.txt"
# TRAINING_PNG_FILE = "nlpir01/2020_2_results/nlpir01.training_acc_loss.png"

PREDICT_FILE_COLUMNS = ['qid', 'Q0', 'docno', 'rank', 'score', 'tag']

LABEL_COLUMN = "label"

NUM_WORDS = 15000
EMBEDDING_SIZE = 200

BATCH_SIZE = 32
EPOCHS = 100
