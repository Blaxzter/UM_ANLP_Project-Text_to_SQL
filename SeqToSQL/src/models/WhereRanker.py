import numpy as np
import torch
from torch import nn, optim
from transformers import BertModel, get_linear_schedule_with_warmup

from utils.Constants import PRE_TRAINED_MODEL_NAME


class WhereRanker(nn.Module):
    def __init__(self):
        super(WhereRanker, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0),
            token_type_ids=token_type_ids.squeeze(0)
        )
        output = self.drop(outputs.pooler_output)
        linear = self.linear(output)
        softmax = torch.softmax(
            torch.sigmoid(linear), dim = 0
        )
        return torch.transpose(softmax, 0, 1)


class WhereRankerTrainer:

    def __init__(self, device, dataset):
        self.where_ranker = WhereRanker().to(device)
        self.loss_function = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.where_ranker.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        where_targets = data["target"]['WHERE'].to(device)
        where_outputs = self.predict(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        self.calc_loss(where_outputs, where_targets)

    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.where_ranker(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        return outputs

    def train(self):
        self.where_ranker = self.where_ranker.train()

    def calc_loss(self, outputs, targets):
        top_where_selection = torch.topk(outputs, k=len(targets), dim=1)[1]
        self.correct_predictions += 1 if top_where_selection == targets else 0
        loss = self.loss_function(outputs, targets.view(-1))
        self.losses.append(loss.item())

        loss.backward()

    def step(self):
        nn.utils.clip_grad_norm_(self.where_ranker.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'Where ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')
