import os
from datetime import datetime
from typing import Dict

import torch
from tqdm import tqdm
from utils.Constants import save_folder

from models.AggregationClassifier import AggregationClassifierTrainer
from models.QABert import QABertTrainer
from models.SelectRanker import SelectRankerTrainer
from models.WhereConditionClassifier import WhereConditionClassifierTrainer
from models.WhereNumberClassifier import WhereNumberClassifierTrainer
from models.WhereRanker import WhereRankerTrainer


def train_epoch(models: Dict, data_loader, device, batch_size=16, report_size = 64):
    # Set the models to train mode
    map(lambda x: x.train(), models)

    sent_cnt = 0

    with tqdm(data_loader, unit="it") as tepoch:
        for d in tepoch:
            sent_cnt += 1
            # Get sentence encoding
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            metrics = ""
            for model_key in models:
                models[model_key].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
                models[model_key].step()
                id, acc, loss = models[model_key].get_metric()
                metrics = metrics + f' {id}[Acc: {acc}, Loss: {loss}], '

                tepoch.set_postfix_str(metrics)
            #models["where_numb_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            #models["selection_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            #models["agg_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            #models["where_ranker_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            #models["where_cond_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            #models['qa_trainer'].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            #if (sent_cnt % batch_size) == 0 or sent_cnt == len(data_loader) - 1:
            #    map(lambda x: x.step(), models)

            #if (sent_cnt % report_size) == 0 or sent_cnt == len(data_loader) - 1:
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
        selection_trainer=SelectRankerTrainer(device, dev_data_loader),
        agg_class_trainer=AggregationClassifierTrainer(device, dev_data_loader),
        where_ranker_trainer=WhereRankerTrainer(device, dev_data_loader),
        where_cond_class_trainer=WhereConditionClassifierTrainer(device, dev_data_loader),
        where_numb_class_trainer=WhereNumberClassifierTrainer(device, dev_data_loader),
        qa_trainer=QABertTrainer(device, dev_data_loader),
    )
    for name in models:
        models[name].get_model().load_state_dict(torch.load(path + "/" + name + ".ckpt"))
        print(f"Loaded {name}")
    return models
