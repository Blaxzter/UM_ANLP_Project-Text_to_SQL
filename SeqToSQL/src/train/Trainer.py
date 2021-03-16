from typing import Dict

from tqdm import tqdm


def train_epoch(models: Dict, data_loader, device, batch_size = 16):

    # Set the models to train mode
    map(lambda x: x.train(), models)

    sent_cnt = 0

    with tqdm(data_loader, unit = "batch") as tepoch:
        for d in tepoch:
            sent_cnt += 1
            # Get sentence encoding
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            token_type_ids = d["token_type_ids"].to(device)

            models["where_numb_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            models["selection_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            models["agg_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)

            models["where_ranker_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            models["where_cond_class_trainer"].train_model_step(d, device, input_ids, attention_mask, token_type_ids)
            models['qa_trainer'].train_model_step(d, device)

            if (sent_cnt % batch_size) == 0 or sent_cnt == len(data_loader) - 1:
                map(lambda x: x.step(), models)

            if (sent_cnt % batch_size) == 0 or sent_cnt == len(data_loader) - 1:
                map(lambda x: x.report_error(), models)

