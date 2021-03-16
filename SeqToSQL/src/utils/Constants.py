from transformers import BertTokenizer

data_folder = '../data'
save_folder = '../checkpoint'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
sep_token = tokenizer.sep_token
cls_token = tokenizer.cls_token

EPOCHS = 3


NUM_AGGREGATIONS = 6
NUM_WHERE_CONDITIONS = 4
NUM_MAX_CONDITIONS = 7
