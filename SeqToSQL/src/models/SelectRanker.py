import numpy as np
import torch
from torch import nn, optim
from transformers import BertModel, get_linear_schedule_with_warmup

from utils.Constants import PRE_TRAINED_MODEL_NAME


class SelectRanker(nn.Module):

    def __init__(self, base_model = None):
        super(SelectRanker, self).__init__()

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
        #softmax = torch.log_softmax(
        #    torch.sigmoid(linear), dim = 0
        #)
        softmax = torch.log_softmax(linear, dim=0)
        return torch.transpose(softmax, 0, 1)


class SelectRankerTrainer:

    def __init__(self, device, dataset, base_model = None):
        self.selection_ranker = SelectRanker(base_model).to(device)
        self.loss_function = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.selection_ranker.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0

        self.train_mode = True

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        select_targets = data["target"]['SELECT'].to(device)
        select_outputs = self.predict(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        return self.calc_loss(select_outputs, select_targets)

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
        outputs = self.selection_ranker(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        return outputs

    def get_prediction(self, input_ids, attention_mask, token_type_ids):
        outputs = self.predict(input_ids, attention_mask, token_type_ids)
        return torch.argmax(outputs, dim = 1)

    def eval(self):
        self.train_mode = False
        self.selection_ranker = self.selection_ranker.eval()

    def train(self):
        self.train_mode = True
        self.selection_ranker = self.selection_ranker.train()

    def calc_loss(self, outputs, targets):
        pred_req_id = torch.argmax(outputs, dim = 1)
        correct_prediction = 1 if pred_req_id == targets else 0

        if self.train_mode:
            self.correct_predictions += correct_prediction
        loss = self.loss_function(outputs, targets.view(-1))

        if self.train_mode:
            self.losses.append(loss.item())
            loss.backward()

        return loss.item(), correct_prediction

    def step(self):
        nn.utils.clip_grad_norm_(self.selection_ranker.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'Select ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')

    def get_metric(self):
        return "SRank", round(self.correct_predictions / len(self.losses), 2), round(np.mean(self.losses), 2)

    def get_model(self):
        return self.selection_ranker
