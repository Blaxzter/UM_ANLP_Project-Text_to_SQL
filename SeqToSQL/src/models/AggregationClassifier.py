import numpy as np
import torch
from torch import nn, optim
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertModel

from utils.Constants import PRE_TRAINED_MODEL_NAME, NUM_AGGREGATIONS


class AggregationClassifier(nn.Module):
    def __init__(self):
        super(AggregationClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, NUM_AGGREGATIONS)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0)
        )
        output = self.drop(outputs.pooler_output)
        linear = self.linear(output)
        softmax = torch.softmax(
            torch.sigmoid(linear), dim = 1
        )
        return softmax


class AggregationClassifierPreTrained(nn.Module):
    def __init__(self):
        super(AggregationClassifierPreTrained, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels = NUM_AGGREGATIONS,
            output_attentions = False,
            output_hidden_states = False,
        )
        self.drop = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0)
        )
        return outputs


class AggregationClassifierTrainer:

    def __init__(self, device, dataset):
        self.agg_classifier = AggregationClassifierPreTrained().to(device)
        self.loss_function = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.agg_classifier.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        agg_target = data["target"]['SELECT_AGG'].to(device)
        agg_output = self.predict(
            input_ids,
            attention_mask,
            token_type_ids,
            agg_target.view(-1)
        )
        self.calc_loss(
            agg_output, agg_target
        )

    def predict(self, input_ids, attention_mask, token_type_ids, selected_column):
        outputs = self.agg_classifier(
            input_ids = input_ids.squeeze(0)[selected_column].view(-1),
            attention_mask = attention_mask.squeeze(0)[selected_column].view(-1),
            token_type_ids = token_type_ids.squeeze(0)[selected_column].view(-1),
        )
        return outputs['logits']

    def train(self):
        self.agg_classifier = self.agg_classifier.train()

    def calc_loss(self, outputs, targets):
        pred_req_id = torch.argmax(outputs, dim = 1)
        self.correct_predictions += 1 if pred_req_id == targets else 0
        loss = self.loss_function(outputs, targets.view(-1))
        self.losses.append(loss.item())

        loss.backward()

    def step(self):
        nn.utils.clip_grad_norm_(self.agg_classifier.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'Aggregation Ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')

    def get_model(self):
        return self.agg_classifier