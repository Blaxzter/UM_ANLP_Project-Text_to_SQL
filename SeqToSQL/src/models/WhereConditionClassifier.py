import numpy as np
import torch
from torch import nn, optim
from transformers import BertModel, get_linear_schedule_with_warmup

from utils.Constants import PRE_TRAINED_MODEL_NAME, NUM_AGGREGATIONS, NUM_WHERE_CONDITIONS


class WhereConditionClassifier(nn.Module):
    def __init__(self):
        super(WhereConditionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p = 0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, NUM_WHERE_CONDITIONS)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids = input_ids.unsqueeze(0),
            attention_mask = attention_mask.unsqueeze(0),
            token_type_ids = token_type_ids.unsqueeze(0)
        )
        output = self.drop(outputs.pooler_output)
        linear = self.linear(output)
        softmax = torch.softmax(
            torch.sigmoid(linear), dim = 1
        )
        return softmax


class WhereConditionClassifierTrainer:

    def __init__(self, device, dataset):
        self.where_cond_classifier = WhereConditionClassifier().to(device)
        self.loss_function = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.where_cond_classifier.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        where_cond_targets = data["target"]['WHERE_CONDITIONS'].to(device)
        where_columns = data["target"]['WHERE'].to(device)
        for where_column, where_cond_target in zip(where_columns, where_cond_targets):
            where_outputs = self.predict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                where_column = where_column
            )

            self.calc_loss(where_outputs, where_cond_target)

    def predict(self, input_ids, attention_mask, token_type_ids, where_column):
        outputs = self.where_cond_classifier(
            input_ids = input_ids.squeeze(0)[where_column].view(-1),
            attention_mask = attention_mask.squeeze(0)[where_column].view(-1),
            token_type_ids = token_type_ids.squeeze(0)[where_column].view(-1),
        )
        return outputs

    def train(self):
        self.where_cond_classifier = self.where_cond_classifier.train()

    def calc_loss(self, outputs, targets):
        pred_req_id = torch.argmax(outputs, dim = 1)
        self.correct_predictions += 1 if pred_req_id == targets else 0
        loss = self.loss_function(outputs, targets.view(-1))
        self.losses.append(loss.item())

        loss.backward()

    def step(self):
        nn.utils.clip_grad_norm_(self.where_cond_classifier.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'Selection Ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')
