import numpy as np
import torch

from torch import nn, optim
from transformers import get_linear_schedule_with_warmup, BertModel, BertForQuestionAnswering

from utils.Constants import PRE_TRAINED_MODEL_NAME


class QABert(nn.Module):
    def __init__(self, output_num_words):
        super(QABert, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p = 0.3)
        self.linearStart = nn.Linear(self.bert.config.hidden_size, output_num_words)
        self.linearEnd = nn.Linear(self.bert.config.hidden_size, output_num_words)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids = input_ids.unsqueeze(0),
            attention_mask = attention_mask.unsqueeze(0),
            token_type_ids = token_type_ids.unsqueeze(0)
        )
        start_values = self.linearStart(outputs.last_hidden_state)
        # start_values_with_attention_mask = torch.mul(attention_mask, start_values.view(-1))
        start_softmax = torch.softmax(start_values, dim = 1)

        end_values = self.linearEnd(outputs.last_hidden_state)
        # end_values_with_attention_mask = torch.mul(attention_mask, end_values.view(-1))
        end_softmax = torch.softmax(end_values, dim = 1)

        return start_softmax, end_softmax

class QABertPreTrained(nn.Module):

    def __init__(self):
        super(QABertPreTrained, self).__init__()
        self.bert = BertForQuestionAnswering.from_pretrained(PRE_TRAINED_MODEL_NAME)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids = input_ids.unsqueeze(0),
            attention_mask = attention_mask.unsqueeze(0),
            token_type_ids = token_type_ids.unsqueeze(0)
        )

        return outputs['start_logits'], outputs['end_logits']

class QABertTrainer:

    def __init__(self, device, dataset):
        self.qa_bert = QABertPreTrained().to(device)

        self.loss_function1 = nn.NLLLoss().to(device)
        self.loss_function2 = nn.NLLLoss().to(device)

        self.optimizer = optim.Adam(self.qa_bert.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.start_losses = []
        self.end_losses = []
        self.correct_predictions = 0

    def train_model_step(self, data, device):
        where_input_ids = data["qa_input_ids"].to(device)
        where_attention_mask = data["qa_attention_mask"].to(device)
        where_token_type_ids = data["qa_token_type_ids"].to(device)
        for cond_num, where_cond_target in enumerate(data["target"]['WHERE_VALUE']):
            start_softmax, end_softmax = self.predict(
                input_ids = where_input_ids.squeeze(0)[cond_num].view(-1),
                attention_mask = where_attention_mask.squeeze(0)[cond_num].view(-1),
                token_type_ids = where_token_type_ids.squeeze(0)[cond_num].view(-1),
            )

            self.calc_loss(start_softmax, end_softmax, where_cond_target)

    def predict(self, input_ids, attention_mask, token_type_ids):
        start_softmax, end_softmax = self.qa_bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        return start_softmax, end_softmax

    def train(self):
        self.qa_bert = self.qa_bert.train()

    def calc_loss(self, start_softmax, end_softmax, targets):
        start_id = torch.argmax(start_softmax)
        end_id = torch.argmax(end_softmax)

        self.correct_predictions += 1 if start_id == targets[0] and end_id == targets[1] else 0

        # todo utilising two losses doesnt work ._.
        start_loss = self.loss_function1(start_softmax, targets[0].view((1, 1)))
        end_loss = self.loss_function2(end_softmax, targets[1].view((1, 1)))

        self.start_losses.append(start_loss.item())
        self.end_losses.append(end_loss.item())

        loss = start_loss + end_loss

        loss.backward()

    def step(self):
        nn.utils.clip_grad_norm_(self.qa_bert.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'QA Question: Correct predictions: {self.correct_predictions / sent_cnt}, mean start loss: {np.mean(self.start_losses)}, mean end loss {np.mean(self.end_losses)}')

    def get_model(self):
        return self.qa_bert