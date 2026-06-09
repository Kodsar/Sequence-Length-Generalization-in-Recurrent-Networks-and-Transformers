from dataclasses import asdict
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics import Accuracy

from configs.palindrome.transformer import config
from models.transformer import TransformerClassifier

class TransformerLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(asdict(config.model))
        self.train_config = config.train
        
        self.model = TransformerClassifier(**asdict(config.model))
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_acc = Accuracy(task="multiclass", num_classes=2)
                
    def forward(self, x, lengths):
        return self.model(x, lengths)

    def training_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        loss = self.criterion(logits, targets)
        
        self.train_acc(logits, targets)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        self.val_acc(logits, targets)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        inputs, targets, lengths = batch
        logits = self(inputs, lengths)
        
        self.test_acc(logits, targets)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), **self.train_config.optimizer)
        return optimizer