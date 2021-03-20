import os
from datetime import datetime
from typing import Dict

import numpy as np
import torch
from tqdm.notebook import tqdm

from dataLoader.DataLoaderUtils import get_question_answers_def_length, get_question_answers, get_qa_embeddings
from utils.Constants import save_folder

from models.AggregationClassifier import AggregationClassifierTrainer
from models.QABert import QABertTrainer
from models.SelectRanker import SelectRankerTrainer
from models.WhereConditionClassifier import WhereConditionClassifierTrainer
from models.WhereNumberClassifier import WhereNumberClassifierTrainer
from models.WhereRanker import WhereRankerTrainer


def train_epoch(models: Dict, train_data_loader, eval_data_loader, device, batch_size = 16, report_size = 8, eval_size = 64, writer = None):
    # Set the models to train mode
    for model in models.values():
        model.train()

    generation = 0

    with tqdm(train_data_loader, desc = "Train", unit = "it") as tepoch:
        for d in tepoch:
            generation += 1

            # Get sentence encoding
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            metrics = ""
            for model_name, model in models.items():
                model.train_model_step(d, device, input_ids, attention_mask, token_type_ids)
                if (generation % batch_size) == 0 or generation == len(train_data_loader) - 1:
                    model.step()
                    id, acc, loss = model.get_metric()
                    metrics = metrics + f' {id}[Acc: {acc}, Loss: {loss}], '


                    # tepoch.set_postfix_str(metrics)
            # if (generation % batch_size) == 0 or generation == len(train_data_loader) - 1:
            #     print(metrics)

            if ((generation % report_size) == 0 or generation == len(train_data_loader) - 1) and writer is not None:

                for model_name, model in models.items():
                    id, acc, loss = model.get_metric()
                    writer.add_scalar(f'Train_Accuracy/{model_name}', acc, generation)
                    writer.add_scalar(f'Train_Loss/{model_name}', loss, generation)

            if (generation % eval_size) == 0 or generation == len(train_data_loader) - 1:

                eval_model(
                    models = models,
                    eval_data_loader = eval_data_loader,
                    generation = generation,
                    device = device,
                    writer = writer
                )

def get_eval(models: Dict, eval_data_loader, max_test_size, device):
    with torch.no_grad():

        for model in models.values():
            model.eval()

        loss_datas = {
            model_name: [] for model_name in models.keys()
        }

        acc_datas = {
            model_name: [] for model_name in models.keys()
        }
        counter = 0

        with tqdm(eval_data_loader, desc = "Eval", unit = "it", leave=False) as eval_tepoch:
            for d in eval_tepoch:
                # Get sentence encoding
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                token_type_ids = d["token_type_ids"].to(device)

                for key, model in models.items():
                    loss, acc = model.train_model_step(d, device, input_ids, attention_mask, token_type_ids)
                    loss_datas[key].append(loss)
                    acc_datas[key].append(acc)

                counter += 1

                if counter > max_test_size:
                    break

    results = {}
    for key, model in models.items():
        loss = np.mean(loss_datas[key])
        acc = np.mean(acc_datas[key])
        results[key] = {"loss": loss, "acc": acc}

    return results


def eval_model(models: Dict, eval_data_loader, generation, device, writer = None):
    with torch.no_grad():

        for model in models.values():
            model.eval()

        eval_data = np.random.choice(list(eval_data_loader), 8)

        loss_datas = {
            model_name: [] for model_name in models.keys()
        }

        acc_datas = {
            model_name: [] for model_name in models.keys()
        }

        with tqdm(eval_data, desc = "Eval", unit = "it", leave=False) as eval_tepoch:
            for d in eval_tepoch:
                # Get sentence encoding
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                token_type_ids = d["token_type_ids"].to(device)

                for key, model in models.items():
                    loss, acc = model.train_model_step(d, device, input_ids, attention_mask, token_type_ids)
                    loss_datas[key].append(loss)
                    acc_datas[key].append(acc)

    for key, model in models.items():
        if writer is not None:
            loss = np.mean(loss_datas[key])
            acc = np.mean(acc_datas[key])
            writer.add_scalar(f'Eval_Accuracy/{key}', acc, generation)
            writer.add_scalar(f'Eval_Loss/{key}', loss, generation)
        model.train()


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


def get_request(models, table_name, columns, types, question, tokenizer, device):
    _req_embeddings = get_question_answers(dict(
        table_name = table_name,
        columns = columns,
        question = question,
        types = types,
    ), tokenizer)

    _input_ids = torch.tensor([req_embedding['input_ids'] for req_embedding in _req_embeddings]).to(device)
    _token_type_ids = torch.tensor([req_embedding['token_type_ids'] for req_embedding in _req_embeddings]).to(device)
    _attention_mask = torch.tensor([req_embedding['attention_mask'] for req_embedding in _req_embeddings]).to(device)

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
        wrc = where_ranker_columns[0][i]
        where_condition = where_cond_class_trainer.get_prediction(_input_ids, _attention_mask, _token_type_ids, wrc)
        wc_idx = where_condition.item()

        column_name = columns[wrc]
        qa_embeds = get_qa_embeddings(tokenizer, question, column_name, wc_idx)
        _qa_input_ids = torch.tensor(qa_embeds['input_ids']).to(device)
        _qa_token_type_ids = torch.tensor(qa_embeds['token_type_ids']).to(device)
        _qa_attention_mask = torch.tensor(qa_embeds['attention_mask']).to(device)
        #_qa_input_ids = torch.tensor([req_embedding['input_ids'] for req_embedding in qa_embeds]).to(device)
        #_qa_token_type_ids = torch.tensor([req_embedding['token_type_ids'] for req_embedding in qa_embeds]).to(device)
        #_qa_attention_mask = torch.tensor([req_embedding['attention_mask'] for req_embedding in qa_embeds]).to(device)
        start_index, end_index = qa_trainer.get_prediction(_qa_input_ids, _qa_token_type_ids, _qa_attention_mask)

        tokens = tokenizer.convert_ids_to_tokens(_qa_input_ids)
        answer = ' '.join(tokens[start_index:end_index + 1])
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

    if aggregation==0:
        agg_col = columns[selected_column] + ' '
    else:
        agg_col = agg_ops[aggregation] + '(' + columns[selected_column] + ') '

    select_part = 'SELECT ' + agg_col + 'FROM ' + table_name

    produced_where_cond = "" if len(where_conditions) == 0 else " WHERE "
    for i, c_w_c in enumerate(where_conditions):
        produced_where_cond += f'{c_w_c["column_name"]} {cond_dict[c_w_c["agg"]]} "{c_w_c["value"]}" '
        if i < len(where_conditions) - 1:
            produced_where_cond += 'AND '

    return select_part + produced_where_cond
