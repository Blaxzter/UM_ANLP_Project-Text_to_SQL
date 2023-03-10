import numpy as np
import torch
from torch import nn, optim
from transformers import BertModel, get_linear_schedule_with_warmup

from utils.Constants import PRE_TRAINED_MODEL_NAME


class WhereRanker(nn.Module):
    def __init__(self, base_model = None):
        super(WhereRanker, self).__init__()
        if base_model is None:
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        else:
            self.bert = base_model
        self.drop = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)


    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            token_type_ids=token_type_ids.squeeze(0)
        )
        output = self.drop(outputs.pooler_output)
        linear = self.linear(output)
        # #softmax = torch.log_softmax(
        # #    torch.sigmoid(linear), dim = 0
        # #)
        # softmax = torch.log_softmax(linear, dim=0)
        sigmoid = torch.sigmoid(linear)
        return torch.transpose(sigmoid, 0, 1)


class WhereRankerTrainer:

    def __init__(self, device, dataset, base_model=None):
        self.where_ranker = WhereRanker(base_model).to(device)
        self.loss_function = nn.BCEWithLogitsLoss().to(device)
        self.optimizer = optim.Adam(self.where_ranker.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0
        self.train_mode = True

    def parse_input(self, d):
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        token_type_ids = d["token_type_ids"]
        return (
            input_ids,
            attention_mask,
            token_type_ids
        )

    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.where_ranker(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        return outputs

    def get_prediction(self, input_ids, attention_mask, token_type_ids, num_where_column):
        outputs = self.predict(input_ids, attention_mask, token_type_ids)
        top_where_selection = torch.topk(outputs, k = num_where_column, dim = 1)[1]
        return top_where_selection

    def eval(self):
        self.train_mode = False
        self.where_ranker = self.where_ranker.eval()

    def train(self):
        self.train_mode = True
        self.where_ranker = self.where_ranker.train()

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        where_targets = data["target"]['WHERE'].to(device)
        where_outputs = self.predict(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        return self.calc_loss(where_outputs, where_targets)

    def calc_loss(self, outputs, targets):
        num_where_columns = torch.count_nonzero(targets).item()
        target_idx = torch.topk(targets, k=num_where_columns, dim=1)[1]
        top_where_selection = torch.topk(outputs, k=num_where_columns, dim=1)[1]

        correct_prediction = 1 if torch.all(top_where_selection == target_idx) else 0
        if self.train_mode:
            self.correct_predictions += correct_prediction
        loss = self.loss_function(outputs, targets)

        if self.train_mode:
            self.losses.append(loss.item())
            loss.backward()

        return loss.item(), correct_prediction

    def step(self):
        nn.utils.clip_grad_norm_(self.where_ranker.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'Where ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')

    def get_metric(self):
        return "WRank", round(self.correct_predictions / len(self.losses), 2), round(np.mean(self.losses), 2)

    def get_model(self):
        return self.where_ranker
