import numpy as np
import torch

from torch import nn, optim
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup, BertModel, BertForQuestionAnswering

from utils.Constants import PRE_TRAINED_MODEL_NAME


class QABert(nn.Module):
    def __init__(self, base_model=None):
        super(QABert, self).__init__()

        if base_model is None:
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        else:
            self.bert = base_model

        self.drop = nn.Dropout(p=0.3)
        self.linearStart = nn.Linear(self.bert.config.hidden_size, 1)
        self.linearEnd = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0)
        )
        start_values = self.linearStart(outputs.last_hidden_state)
        start_values = start_values.view(1, start_values.shape[1])
        # start_values_with_attention_mask = torch.mul(attention_mask, start_values.view(-1))
        start_softmax = torch.log_softmax(start_values, dim=1)

        end_values = self.linearEnd(outputs.last_hidden_state)
        end_values = end_values.view(1, end_values.shape[1])
        # end_values_with_attention_mask = torch.mul(attention_mask, end_values.view(-1))
        end_softmax = torch.log_softmax(end_values, dim=1)

        return start_softmax, end_softmax


class QABertPreTrained(nn.Module):

    def __init__(self):
        super(QABertPreTrained, self).__init__()
        self.bert = BertForQuestionAnswering.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            token_type_ids=token_type_ids.unsqueeze(0)
        )

        return outputs['start_logits'], outputs['end_logits']


class QABertTrainer:

    def __init__(self, device, dataset, base_model=None, use_pretrained=True):
        self.use_pretrained = use_pretrained
        if use_pretrained:
            self.qa_bert = QABertPreTrained().to(device)
        else:
            self.qa_bert = QABert(base_model).to(device)

        self.loss_function1 = nn.NLLLoss().to(device)
        self.loss_function2 = nn.NLLLoss().to(device)

        self.optimizer = optim.Adam(self.qa_bert.parameters(), lr=0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0

    def train_model_step(self, data, device, input_ids, attention_mask, token_type_ids):
        where_input_ids = data["qa_input_ids"].to(device)
        where_attention_mask = data["qa_attention_mask"].to(device)
        where_token_type_ids = data["qa_token_type_ids"].to(device)
        for cond_num, where_cond_target in enumerate(data["target"]['WHERE_VALUE']):
            target_0 = where_cond_target[0].to(device)
            target_1 = where_cond_target[1].to(device)

            start_softmax, end_softmax = self.predict(
                input_ids=where_input_ids.squeeze(0)[cond_num].view(-1),
                attention_mask=where_attention_mask.squeeze(0)[cond_num].view(-1),
                token_type_ids=where_token_type_ids.squeeze(0)[cond_num].view(-1)
            )

            self.calc_loss(start_softmax, end_softmax, target_0, target_1)

    def parse_input(self, d):
        input_ids = d["qa_input_ids"]
        attention_mask = d["qa_attention_mask"]
        token_type_ids = d["qa_token_type_ids"]
        return (
            input_ids.squeeze(0)[0].view(-1),
            attention_mask.squeeze(0)[0].view(-1),
            token_type_ids.squeeze(0)[0].view(-1),
        )

    def predict(self, input_ids, attention_mask, token_type_ids):
        start_softmax, end_softmax = self.qa_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return start_softmax, end_softmax

    def train(self):
        self.qa_bert = self.qa_bert.train()

    def calc_loss(self, start_logits, end_logits, start_positions, end_positions):
        start_id = torch.argmax(start_logits)
        end_id = torch.argmax(end_logits)

        self.correct_predictions += 1 if start_id == start_positions and end_id == end_positions else 0

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

        self.losses.append(total_loss.item())

        total_loss.backward()

    def step(self):
        nn.utils.clip_grad_norm_(self.qa_bert.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(
            f'QA Question: Correct predictions: {self.correct_predictions / sent_cnt}, mean start loss: {np.mean(self.start_losses)}, mean end loss {np.mean(self.end_losses)}')

    def get_metric(self):
        return "QAB", round(self.correct_predictions / len(self.losses), 2), round(np.mean(self.losses), 2)

    def get_model(self):
        return self.qa_bert
