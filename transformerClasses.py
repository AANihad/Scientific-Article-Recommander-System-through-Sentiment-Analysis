import pytorch_lightning as pl
from transformers import AutoModel, get_cosine_schedule_with_warmup
from transformers import XLNetModel

import torch
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
attributes = ['Positive', 'Objective','Negative']

class Citation_Classifier(pl.LightningModule):
  def __init__(self, xlnet, config: dict):
    super().__init__()
    self.config = config
    if xlnet:
      self.pretrained_model = XLNetModel.from_pretrained(config['model_name'], return_dict = True, num_labels =  self.config['n_labels'])
    else:
      self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
    self.hidden = nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
    self.classifier = nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
    torch.nn.init.xavier_uniform_(self.classifier.weight)
    torch.nn.init.xavier_uniform_(self.hidden.weight)
    self.loss_func = nn.CrossEntropyLoss(weight = config['weights'])
    self.dropout = nn.Dropout()
    self.soft = nn.Softmax(dim=1)
   
  def forward(self, input_ids, attention_mask, labels=None):
    # scibert
    output = self.pretrained_model(input_ids=input_ids, attention_mask = attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = self.dropout(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output) 
    #logits =  self.soft(pooled_output)
    # calculate loss
    loss = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
    return loss, logits

  def training_step(self, batch, batch_index):
    # loss, outputs = self(**batch)
    # self.log("train loss ", loss, prog_bar = True, logger=True)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_index):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def predict_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    return outputs

  def training_epoch_end(self, outputs):
    
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    for i, name in enumerate(attributes):
      class_roc_auc = auroc(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)


  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    total_steps = self.config['train_size']/self.config['batch_size'] # The n training step
    warmup_steps = math.floor(total_steps * self.config['warmup'])

    #'warmup': 32, # (len(Data_module.train_dataloader())/4)/5 or tot steps * 0.2

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]

def getConfig(model_name):
    return {
    'model_name': model_name,
    'n_labels': 3,
    'batch_size': 32,
    'lr': 1.5e-5,
    'warmup': 0.2, 
    'train_size': 1,
    'weight_decay': 0.00,
    'n_epochs': 4,
    'weights' : torch.tensor([0.7/3, 0.1/3, 2.2/3], dtype=torch.float)
}
