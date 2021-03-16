import torch

from torch import nn, optim
from transformers import get_linear_schedule_with_warmup, BertModel

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
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        start_values = self.linearStart(outputs.last_hidden_state)
        # start_values_with_attention_mask = torch.mul(attention_mask, start_values.view(-1))
        start_softmax = torch.softmax(start_values, dim = 1)

        end_values = self.linearEnd(outputs.last_hidden_state)
        # end_values_with_attention_mask = torch.mul(attention_mask, end_values.view(-1))
        end_softmax = torch.softmax(end_values, dim = 1)

        return start_softmax, end_softmax


class QABertTrainer:

    def __init__(self, device, dataset):
        self.qa_bert = QABert(1).to(device)
        self.loss_function = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.qa_bert.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0

    def train_model_step(self, data, device):
        where_input_ids = data["qa_input_ids"].to(device)
        where_attention_mask = data["qa_attention_mask"].to(device)
        where_token_type_ids = data["qa_token_type_ids"].to(device)
        for cond_num, where_cond_target in enumerate(data["target"]['WHERE_VALUE']):
            start_softmax, end_softmax = self.predict(
                input_ids = where_input_ids[cond_num],
                attention_mask = where_attention_mask[cond_num],
                token_type_ids = where_token_type_ids[cond_num],
            )

            self.calc_loss(start_softmax, end_softmax, where_cond_target.view(-1))

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
        start_id = torch.argmax(start_softmax, dim = 1)
        end_id = torch.argmax(end_softmax, dim = 1)

        self.correct_predictions += 1 if start_id == targets[0] and end_id == targets[1] else 0

        # todo utilising two losses doesnt work ._.
        start_loss = self.loss_function(start_softmax, targets[0].view((1, 1)))
        end_loss = self.loss_function(end_softmax, targets[1].view((1, 1)))

        self.losses.append(start_loss.item())
        self.losses.append(end_loss.item())

        start_loss.backward()
        end_loss.backward()

    def step(self):
        nn.utils.clip_grad_norm_(self.qa_bert.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def report_error(self, sent_cnt):
        print(f'Selection Ranker: Correct predictions: {self.correct_predictions / sent_cnt}, mean loss: {np.mean(self.losses)}')
