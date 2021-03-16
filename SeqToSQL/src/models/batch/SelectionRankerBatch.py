import torch
from torch import nn, optim
from transformers import get_linear_schedule_with_warmup, BertModel

from utils.Constants import PRE_TRAINED_MODEL_NAME


class SelectionRankerBatch(nn.Module):
    def __init__(self):
        super(SelectionRankerBatch, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p = 0.3)
        self.linear = nn.Linear(self.bert.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.bert(
            input_ids=input_ids.flatten(end_dim=1),
            attention_mask=attention_mask.flatten(end_dim=1),
            token_type_ids=token_type_ids.flatten(end_dim=1)
        )

        output = self.drop(outputs.pooler_output)
        linear = self.linear(output.view(16, 5, 768))
        softmax = torch.softmax(
            torch.sigmoid(linear),
            dim = 1
        )
        return softmax


class SelectTrainerBatch:
    def __init__(self, device, dataset):
        self.selection_ranker_batch = SelectionRankerBatch()
        self.loss_function = nn.NLLLoss().to(device)
        self.optimizer = optim.Adam(self.selection_ranker_batch.parameters(), lr = 0.01)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps = 0,
            num_training_steps = len(dataset)
        )
        self.num_examples = len(dataset)
        self.losses = []
        self.correct_predictions = 0

    def predict(self, input_ids, attention_mask, token_type_ids):
        outputs = self.selection_ranker_batch(
            input_ids = input_ids,
            attention_masks = attention_mask,
            token_type_ids = token_type_ids
        )
        return outputs

    def train(self):
        return self.selection_ranker_batch.train()

    def step(self, outputs, targets):
        self.correct_predictions += torch.sum(torch.argmax(outputs, dim = 1) == targets.squeeze(1))
        loss = self.loss_function(outputs, targets.squeeze(1))
        self.losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(self.selection_ranker_batch.parameters(), max_norm = 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

