import numpy as np
import torch
from torch import nn, optim
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertModel

from utils.Constants import PRE_TRAINED_MODEL_NAME, NUM_AGGREGATIONS


class AggregationClassifier(nn.Module):
    def __init__(self, base_model=None):
        super(AggregationClassifier, self).__init__()
        if base_model is None:
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        else:
            self.bert = base_model
        self.drop = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, NUM_AGGREGATIONS)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0)
        )
        output = self.drop(outputs.pooler_output)
        linear = self.linear(output)
        softmax = torch.log_softmax(
            linear, dim=1
        )
        return softmax


class AggregationClassifierPreTrained(nn.Module):
    def __init__(self):
        super(AggregationClassifierPreTrained, self).__init__()

        self.bert = BertForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels=NUM_AGGREGATIONS,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.drop = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0)
        )
        return outputs['logits']


class AggregationClassifierTrainer:

    def __init__(self, device, dataset, base_model=None, use_pretrained=True):
        self.use_pretrained = use_pretrained
        if self.use_pretrained:
            self.agg_classifier = AggregationClassifierPreTrained().to(device)
        else:
            self.agg_classifier = AggregationClassifier(base_model).to(device)

        self.loss_function = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.agg_classifier.parameters(), lr=0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0
        self.train_mode = True

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        agg_target = data["target"]['SELECT_AGG'].to(device)
        select_target = data["target"]['SELECT'].to(device)
        agg_output = self.predict(
            input_ids,
            attention_mask,
            token_type_ids,
            select_target.view(-1)
        )
        return self.calc_loss(
            agg_output, agg_target
        )

    def parse_input(self, d):
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        token_type_ids = d["token_type_ids"]
        agg_target = d["target"]['SELECT_AGG']
        return (
            input_ids.squeeze(0)[agg_target.view(-1)].view(-1),
            attention_mask.squeeze(0)[agg_target.view(-1)].view(-1),
            token_type_ids.squeeze(0)[agg_target.view(-1)].view(-1),
        )

    def predict(self, input_ids, attention_mask, token_type_ids, selected_column):
        outputs = self.agg_classifier(
            input_ids=input_ids.squeeze(0)[selected_column].view(-1),
            attention_mask=attention_mask.squeeze(0)[selected_column].view(-1),
            token_type_ids=token_type_ids.squeeze(0)[selected_column].view(-1),
        )

        return outputs

    def get_prediction(self, input_ids, attention_mask, token_type_ids, selected_column):
        outputs = self.predict(input_ids, attention_mask, token_type_ids, selected_column)
        return torch.argmax(outputs, dim=1)

    def eval(self):
        self.train_mode = False
        self.agg_classifier.eval()

    def train(self):
        self.train_mode = True
        self.agg_classifier.train()

    def calc_loss(self, outputs, targets):
        pred_req_id = torch.argmax(outputs, dim=1)

        correct_prediction = 1 if pred_req_id == targets else 0
        if self.train_mode:
            self.correct_predictions += correct_prediction

        loss = self.loss_function(outputs, targets.view(-1))

        if self.train_mode:
            self.losses.append(loss.item())
            loss.backward()

        return loss.item(), correct_prediction

    def step(self):
        nn.utils.clip_grad_norm_(self.agg_classifier.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(
            f'Aggregation Ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')

    def get_metric(self):
        return "SAC", round(self.correct_predictions / len(self.losses), 2), round(np.mean(self.losses), 2)

    def get_model(self):
        return self.agg_classifier