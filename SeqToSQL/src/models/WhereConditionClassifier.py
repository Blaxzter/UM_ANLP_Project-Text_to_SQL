import numpy as np
import torch
from torch import nn, optim
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertModel

from utils.Constants import PRE_TRAINED_MODEL_NAME, NUM_AGGREGATIONS, NUM_WHERE_CONDITIONS


class WhereConditionClassifier(nn.Module):
    def __init__(self, base_model = None):
        super(WhereConditionClassifier, self).__init__()
        if base_model is None:
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        else:
            self.bert = base_model

        self.drop = nn.Dropout(p = 0.1)
        self.linear = nn.Linear(self.bert.config.hidden_size, NUM_WHERE_CONDITIONS)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids = input_ids.unsqueeze(0),
            attention_mask = attention_mask.unsqueeze(0),
            token_type_ids = token_type_ids.unsqueeze(0)
        )
        output = self.drop(outputs.pooler_output)
        linear = self.linear(output)
        softmax = torch.log_softmax(
            linear, dim = 1
        )
        return softmax


class WhereConditionClassifierPreTrained(nn.Module):
    def __init__(self):
        super(WhereConditionClassifierPreTrained, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels = NUM_WHERE_CONDITIONS,
            output_attentions = False,
            output_hidden_states = False,
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids = input_ids.unsqueeze(0),
            attention_mask = attention_mask.unsqueeze(0),
            token_type_ids = token_type_ids.unsqueeze(0)
        )
        return outputs['logits']


class WhereConditionClassifierTrainer:

    def __init__(self, device, dataset, base_model=None, use_pretrained=True):
        self.use_pretrained = use_pretrained
        if self.use_pretrained:
            self.where_cond_classifier = WhereConditionClassifierPreTrained().to(device)
        else:
            self.where_cond_classifier = WhereConditionClassifier(base_model).to(device)

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
        self.train_mode = True

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        where_cond_targets = data["target"]['WHERE_CONDITIONS'].to(device)
        where_columns = data["target"]['WHERE'].to(device)
        num_where_columns = torch.count_nonzero(where_columns).item()
        target_idx = torch.topk(where_columns, k=num_where_columns, dim=1)[1].to(device)

        c_losses = []
        c_correct_predictions = []

        for where_column, where_cond_target in zip(target_idx.view(-1), where_cond_targets.view(-1)):
            where_outputs = self.predict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                where_column = where_column
            )

            loss, correct_prediction = self.calc_loss(where_outputs, where_cond_target)
            c_losses.append(loss)
            c_correct_predictions.append(correct_prediction)

        return np.mean(c_losses), np.mean(c_correct_predictions)

    def parse_input(self, d):
        input_ids = d["input_ids"]
        attention_mask = d["attention_mask"]
        token_type_ids = d["token_type_ids"]
        where_columns = d["target"]['WHERE']
        num_where_columns = torch.count_nonzero(where_columns).item()
        target_idx = torch.topk(where_columns, k=num_where_columns, dim=1)[1]
        return (
            input_ids.squeeze(0)[target_idx[0]].view(-1),
            attention_mask.squeeze(0)[target_idx[0]].view(-1),
            token_type_ids.squeeze(0)[target_idx[0]].view(-1),
        )

    def get_prediction(self, input_ids, attention_mask, token_type_ids, where_column):
        outputs = self.predict(
                input_ids = input_ids,
                attention_mask = attention_mask,
                token_type_ids = token_type_ids,
                where_column = where_column
            )
        return torch.argmax(outputs, dim = 1)

    def predict(self, input_ids, attention_mask, token_type_ids, where_column):
        outputs = self.where_cond_classifier(
            input_ids = input_ids.squeeze(0)[where_column].view(-1),
            attention_mask = attention_mask.squeeze(0)[where_column].view(-1),
            token_type_ids = token_type_ids.squeeze(0)[where_column].view(-1),
        )
        return outputs

    def eval(self):
        self.train_mode = False
        self.where_cond_classifier.eval()

    def train(self):
        self.train_mode = True
        self.where_cond_classifier.train()

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
        nn.utils.clip_grad_norm_(self.where_cond_classifier.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'Where condition Ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')

    def get_metric(self):
        return "WCC", round(self.correct_predictions / len(self.losses), 2), round(np.mean(self.losses), 2)

    def get_model(self):
        return self.where_cond_classifier
