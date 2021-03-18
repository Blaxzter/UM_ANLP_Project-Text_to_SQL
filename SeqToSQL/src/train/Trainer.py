import os
from datetime import datetime
from typing import Dict

import torch
from tqdm import tqdm

from dataLoader.DataLoaderUtils import get_question_answers_def_length, get_question_answers, get_qa_embeddings
from utils.Constants import save_folder

from models.AggregationClassifier import AggregationClassifierTrainer
from models.QABert import QABertTrainer
from models.SelectRanker import SelectRankerTrainer
from models.WhereConditionClassifier import WhereConditionClassifierTrainer
from models.WhereNumberClassifier import WhereNumberClassifierTrainer
from models.WhereRanker import WhereRankerTrainer


def train_epoch(models: Dict, data_loader, device, batch_size = 16, report_size = 64, writer = None):
    # Set the models to train mode
    map(lambda x: x.train(), models)

    sent_cnt = 0

    with tqdm(data_loader, unit = "it") as tepoch:
        for d in tepoch:
            sent_cnt += 1

            # Get sentence encoding
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            metrics = ""
            for model_key in models:
                models[model_key].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
                if (sent_cnt % batch_size) == 0 or sent_cnt == len(data_loader) - 1:
                    models[model_key].step()
                    id, acc, loss = models[model_key].get_metric()
                    metrics = metrics + f' {id}[Acc: {acc}, Loss: {loss}], '

                    tepoch.set_postfix_str(metrics)
            # models["where_numb_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            # models["selection_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            # models["agg_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            # models["where_ranker_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            # models["where_cond_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            # models['qa_trainer'].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            # if (sent_cnt % batch_size) == 0 or sent_cnt == len(data_loader) - 1:
            #    map(lambda x: x.step(), models)

            # if (sent_cnt % report_size) == 0 or sent_cnt == len(data_loader) - 1:
            #    map(lambda x: x.report_error(), models)


def save_model(models: Dict, path):
    dateTimeObj = datetime.now()
    subdir = dateTimeObj.strftime("%d_%b_%Y_%H_%M")
    print(f"Subpath: {subdir}")
    for name in models:
        model = models[name].get_model()
        if not os.path.exists(path + "/" + subdir):
            os.makedirs(path + "/" + subdir)
        final_path = path + "/" + subdir + "/" + name + ".ckpt"
        print(f"Saving {name} under {final_path}")
        torch.save(model.state_dict(), final_path)


def load_model(path, dev_data_loader, device):
    models = dict(
        selection_trainer = SelectRankerTrainer(device, dev_data_loader),
        agg_class_trainer = AggregationClassifierTrainer(device, dev_data_loader),
        where_ranker_trainer = WhereRankerTrainer(device, dev_data_loader),
        where_cond_class_trainer = WhereConditionClassifierTrainer(device, dev_data_loader),
        where_numb_class_trainer = WhereNumberClassifierTrainer(device, dev_data_loader),
        qa_trainer = QABertTrainer(device, dev_data_loader),
    )
    for name in models:
        models[name].get_model().load_state_dict(torch.load(path + "/" + name + ".ckpt"))
        print(f"Loaded {name}")
    return models


def get_request(models, table_name, columns, types, question, tokenizer):
    _req_embeddings = get_question_answers(dict(
        table_name = table_name,
        columns = columns,
        question = question,
        types = types,
    ), tokenizer)

    _input_ids = [req_embedding['input_ids'] for req_embedding in _req_embeddings]
    _token_type_ids = [req_embedding['token_type_ids'] for req_embedding in _req_embeddings]
    _attention_mask = [req_embedding['attention_mask'] for req_embedding in _req_embeddings]

    selection_trainer: SelectRankerTrainer = models['selection_trainer']
    agg_class_trainer: AggregationClassifierTrainer = models['agg_class_trainer']

    selected_column = selection_trainer.get_prediction(_input_ids, _attention_mask, _token_type_ids)
    aggregation = agg_class_trainer.get_prediction(_input_ids, _attention_mask, _token_type_ids, selected_column)

    where_numb_class_trainer: WhereNumberClassifierTrainer = models['where_numb_class_trainer']
    where_ranker_trainer: WhereRankerTrainer = models['where_ranker_trainer']
    where_cond_class_trainer: WhereConditionClassifierTrainer = models['where_cond_class_trainer']
    qa_trainer: QABertTrainer = models['qa_trainer']

    where_conditions = []

    numb_where_classes = where_numb_class_trainer.get_prediction(_input_ids, _attention_mask, _token_type_ids)
    where_ranker_columns = where_ranker_trainer.get_prediction(_input_ids, _attention_mask, _token_type_ids, numb_where_classes)

    for i in range(numb_where_classes):
        where_condition = where_cond_class_trainer.get_prediction(_input_ids, _attention_mask, _token_type_ids, where_ranker_columns[i])
        column_name = columns[where_ranker_columns[i]]
        qa_embeds = get_qa_embeddings(tokenizer, question, column_name, where_condition)
        _qa_input_ids = [req_embedding['input_ids'] for req_embedding in qa_embeds]
        _qa_token_type_ids = [req_embedding['token_type_ids'] for req_embedding in qa_embeds]
        _qa_attention_mask = [req_embedding['attention_mask'] for req_embedding in qa_embeds]
        start_index, end_index = qa_trainer.get_prediction(_qa_input_ids, _qa_token_type_ids, _qa_attention_mask)

        tokens = tokenizer.convert_ids_to_tokens(_input_ids[0])
        answer = ' '.join(tokens[start_index:end_index + 1])
        print(answer)
        corrected_answer = ''

        for word in answer.split():
            # If it's a subword token
            if word[0:2] == '##':
                corrected_answer += word[2:]
            else:
                corrected_answer += ' ' + word

        where_conditions.append(dict(
            column_name = column_name,
            agg = where_condition,
            value = corrected_answer,
        ))

    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_dict = ['=', '>', '<', 'OP']

    produced_where_cond = "" if len(where_conditions) == 0 else "WHERE "
    for c_w_c in where_conditions:
        produced_where_cond += f'{c_w_c["column_name"]} {cond_dict[c_w_c["agg"]]} {c_w_c["value"]} '

    return f'SELECT {agg_ops[aggregation]} {columns[selected_column]}\n' \
           f'FROM {table_name}\n' \
           f'{produced_where_cond}'
