import torch

from dataLoader.DataLoader import get_data_loader
from utils.Constants import tokenizer

from models.QABert import QABertTrainer
from models.SelectRanker import SelectRankerTrainer
from models.WhereRanker import WhereRankerTrainer
from models.WhereConditionClassifier import WhereConditionClassifierTrainer
from models.AggregationClassifier import AggregationClassifierTrainer
from train.Trainer import train_epoch

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print('Current device:', torch.cuda.get_device_name(device))
    else:
        print('Failed to find GPU. Will use CPU.')
        device = 'cpu'

    dev_data_loader = get_data_loader(data_type = 'dev', tokenizer = tokenizer, batch_size = 1)

    models = dict(
        selection_trainer = SelectRankerTrainer(device, dev_data_loader),
        agg_class_trainer = AggregationClassifierTrainer(device, dev_data_loader),
        where_ranker_trainer = WhereRankerTrainer(device, dev_data_loader),
        where_cond_class_trainer = WhereConditionClassifierTrainer(device, dev_data_loader),
        qa_trainer = QABertTrainer(device, dev_data_loader),
    )

    train_epoch(models, dev_data_loader, device)
