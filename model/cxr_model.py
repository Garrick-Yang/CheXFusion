import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC, F1Score, CalibrationError
from transformers import get_cosine_schedule_with_warmup
from model.layers import Backbone, FusionBackbone
from model.loss import get_loss


class CxrModel(pl.LightningModule):
    def __init__(self, lr, classes, loss_init_args, timm_init_args):
        super(CxrModel, self).__init__()
        self.lr = lr
        self.classes = classes
        self.backbone = FusionBackbone(timm_init_args)
        # self.backbone = Backbone(timm_init_args)
        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary", average=None)
        self.val_ece = CalibrationError(task="binary", n_bins=15)
        self.criterion_cls = get_loss(**loss_init_args)

    def forward(self, image):
        return self.backbone(image)

    def shared_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)

        loss = self.criterion_cls(pred, label)

        pred = torch.sigmoid(pred).detach()

        return dict(
            loss=loss,
            pred=pred,
            label=label,
        )

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'loss': res['loss'].detach()}, prog_bar=True)
        self.log_dict({'train_loss': res['loss'].detach()}, prog_bar=True, on_step=False, on_epoch=True)
        return res['loss']

    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'val_loss': res['loss'].detach()}, prog_bar=True)
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        num_classes = 40

        preds = torch.cat([x['pred'] for x in self.validation_step_outputs])
        labels = torch.cat([x['label'] for x in self.validation_step_outputs])

        val_ap = []
        val_auroc = []
        val_f1 = []
        val_ece = []
        for i in range(num_classes):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            f1 = self.val_f1(preds[:, i], labels[:, i].long())
            ece = self.val_ece(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            val_f1.append(f1)
            val_ece.append(ece)
            print(
                f'{self.classes[i]}:ap: {ap:.5f}, auroc: {auroc:.5f},f1: {f1:.5f}, ece: {ece:.5f}')

        head_idx = [37, 21, 4, 25, 1, 29, 7, 24, 9, 6]
        medium_idx = [31, 12, 17, 34, 23, 22, 3, 14, 8, 0, 38, 27, 11, 13, 36, 20, 39, 32]
        tail_idx = [10, 33, 28, 16, 15, 30, 18, 26, 2, 19, 5, 35]

        self.log_dict({'val_ap': sum(val_ap) / num_classes}, prog_bar=True)
        self.log_dict({'val_auroc': sum(val_auroc) / num_classes}, prog_bar=False)
        self.log_dict({'val_f1': sum(val_f1) / num_classes}, prog_bar=True)
        self.log_dict({'val_ece': sum(val_ece) / num_classes}, prog_bar=True)
        self.log_dict({'val_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)}, prog_bar=False)
        self.log_dict({'val_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)}, prog_bar=False)
        self.log_dict({'val_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)}, prog_bar=False)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        pred = self.shared_step(batch, batch_idx)['pred']
        image, label = batch
        batch_1 = (image.flip(-1), label)
        pred_1 = self.shared_step(batch_1, batch_idx)['pred']
        pred = (pred + pred_1) / 2
        return pred

    def configure_optimizers(self):
        optimizer = AdamW(self.backbone.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 250000)
        return [optimizer], [scheduler]
