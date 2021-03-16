import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataLoader.DataLoaderUtils import get_question_answers_for_where_value_def_length, get_question_answers_def_length
from utils.Constants import data_folder
from utils.DataUtils import read_json_data_from_file, convert_to_id_dict, get_table_column


class WikiSQLDataset(Dataset):

    def __init__(self, requests, tokenizer, pad_length):
        self.requests = requests
        self.tokenizer = tokenizer
        self.pad_length = pad_length

        self.req_prepared = []

        self.QA_requests = []

        for req in tqdm(requests):
            _qa_values, _qa_targets, _qa_num_cond = get_question_answers_for_where_value_def_length(req, self.tokenizer, self.pad_length)
            _qa_input_ids = [req_embedding['input_ids'] for req_embedding in _qa_values]
            _qa_token_type_ids = [req_embedding['token_type_ids'] for req_embedding in _qa_values]
            _qa_attention_mask = [req_embedding['attention_mask'] for req_embedding in _qa_values]

            _qa_where_value = torch.tensor(_qa_targets)
            _qa_where_num_conditions = torch.tensor(_qa_num_cond)

            _req_embeddings = get_question_answers_def_length(req, self.tokenizer, self.pad_length)
            _input_ids = [req_embedding['input_ids'] for req_embedding in _req_embeddings]
            _token_type_ids = [req_embedding['token_type_ids'] for req_embedding in _req_embeddings]
            _attention_mask = [req_embedding['attention_mask'] for req_embedding in _req_embeddings]

            select_target = torch.tensor([req['sql']['sel']], dtype=torch.long)
            where_target = torch.tensor([cond[0] for cond in req['sql']['conds']], dtype=torch.long)
            where_conditions_target = torch.tensor([cond[1] for cond in req['sql']['conds']], dtype=torch.long)
            select_agg_target = torch.tensor([req['sql']['agg']], dtype=torch.long)

            self.req_prepared.append(dict(
                input_ids = torch.tensor(_input_ids),
                token_type_ids = torch.tensor(_token_type_ids),
                attention_mask = torch.tensor(_attention_mask),

                qa_input_ids = torch.tensor(_qa_input_ids),
                qa_attention_mask = torch.tensor(_qa_attention_mask),
                qa_token_type_ids = torch.tensor(_qa_token_type_ids),

                target = dict(
                    SELECT = select_target,
                    SELECT_AGG = select_agg_target,
                    WHERE = where_target,
                    WHERE_CONDITIONS = where_conditions_target,
                    WHERE_NUM_CONDITIONS = _qa_where_num_conditions ,
                    WHERE_VALUE = _qa_targets
                )
            ))

    def get_full_request_by_id(self, req_id):
        return self.requests[req_id]

    def __len__(self):
        return len(self.req_prepared)

    def __getitem__(self, item):
        return self.req_prepared[item]


def get_data_loader(data_type, tokenizer, batch_size, filter_data = True, pad_length = 65):
    # TODO check if we can use dataLoader with batch size as done in the tutorial
    loaded_req = read_json_data_from_file(f'{data_folder}/{data_type}.jsonl')
    loaded_tables = read_json_data_from_file(f'{data_folder}/{data_type}.tables.jsonl')
    table_data_dict = convert_to_id_dict(loaded_tables, 'id')

    prep_req_data = get_table_column(loaded_req, table_data_dict)

    # if filter_data:
    #     prep_req_data = list(filter(lambda request: len(request['columns']) == 5, prep_req_data))

    print(f'We have {len(loaded_req)} {data_type} data with {len(loaded_tables)} tables.')

    return DataLoader(
        WikiSQLDataset(requests = prep_req_data, tokenizer = tokenizer, pad_length = pad_length),
        batch_size=batch_size
    )
