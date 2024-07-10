from scipy.sparse.construct import random
import torch 
import torch.nn as nn
import torch.functional as F
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from einops import rearrange, repeat
from torch.utils.data import DataLoader, TensorDataset, Dataset

from preprocessing import dataset_prepare
from sklearn.model_selection import KFold


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 10
NUM_GPUS = 1
N_SUBJECT = 1
LR = 1e-4

class ChannelWiseAttention(nn.Module):
    def __init__(self, n_channels, hidden_size):
        super().__init__()
        # first projection layer, bias is True by default 
        self.linear_in = nn.Linear(n_channels, hidden_size)
        # output projection layer, bias is True by default
        self.linear_out = nn.Linear(hidden_size, n_channels)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
    
    def forward(self, x):
        # input size: (batch_size, n_steps, n_channel, n_samples)
        # [256, 3, 32, 128]
        # calculate attention map
        # s_shape = (batch_size, n_steps, n_channel)
        s = torch.mean(x, dim = -1)
        # s_shape = (batch_size, n_steps, hidden_size)
        s = self.linear_in(s)
        # s_shape = (batch_size, n_steps, n_channel)
        s = self.tanh(self.linear_out(s))
        attention_map = self.softmax(s)
        # attention_map shape = (batch_size, n_steps, n_channel, 1)
        attention_map = attention_map.unsqueeze(-1)
        # calculate output: cj = aj * xj
        out = x * attention_map
        # [256, 3, 32, 128]
        return out

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels = in_channels, # in_channels = 1
            out_channels = 40, # 卷积核的数量
            kernel_size = (32, 40), 
        )
        self.elu = nn.ELU()
        self.conv1d = nn.Conv1d(
            in_channels = 40, 
            out_channels = 40, kernel_size = 10)  # 输入通道数为特征向量的维度
        self.pool = nn.MaxPool2d(kernel_size = (1, 75), stride = 10)
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        # input shape: (batch_size, n_steps, n_channels, n_samples)
        # [256, 3, 32, 128]

        batch, step, h, w = x.shape
        x = x.view(batch * step, 1, h, w) 
        # reshape to do parallel convolution on every step
        # new_shape = (batch * n_step, 1, n_channels, n_samples)
        # [768, 1, 32, 128]

        x = self.conv2d(x)
        # after conv [768, 40, 1, 89]

        x = self.elu(x)
        # after elu ([768, 40, 1, 89])

        x = x.squeeze(2)
        # after squeeze ([768, 40, 89])

        x = self.conv1d(x)
        print("after conv1d", x.shape)
        # after conv [768, 40, 80]

        x = x.unsqueeze(2)
        # after unsqueeze ([768, 40, 1, 80])

        x = self.elu(x)
        # after elu ([768, 40, 1, 80])

        x = self.pool(x)
        print("after pool", x.shape)
        # after pool ([768, 40, 1, 2])

        x = self.dropout(x)

        out = x.view(batch, step, -1) 
        # x.view(batch, step, -1) ([256, 3, 80])

        return out

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        # 256 3 1
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out

class RNN_Attention(nn.Module):
    def __init__(self, hidden_size = 64, embed_dim = 64, num_layers = 2, num_heads = 1, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.biLstm = nn.LSTM(
            input_size=80, 
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout,
            bidirectional=True,
        )
        self.self_attention = Attention(
            dim = hidden_size * 2, 
            heads = num_heads,
            dim_head = embed_dim,
            dropout=dropout, 
    
            )

    def forward(self, x):
         # initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE) # (1, batch_size, 128)
        # initialize cell state for LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(DEVICE) # (1, batch_size, 128)

        r_out, _ = self.biLstm(x, (h0, c0))

        output =  self.self_attention(r_out)

        return output

class ARCNN_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_valence = nn.CrossEntropyLoss()
        self.loss = nn.CrossEntropyLoss()
    
    def get_labels(self, preds, target):

        # target: torch.Size([256, 1])
        # preds: torch.Size([256, 3, 2])
        # labels = target.unsqueeze(-1)
        labels = target.expand(preds.size()[0], preds.size()[1])
        # labels = target.unsqueeze(-1).repeat(1, preds.size()[1])
        # print(labels.shape)
        return labels
    
    def __call__(self, preds, target_is_real):
        losses = 0
        labels = self.get_labels(preds, target_is_real)
    
        for i in range(preds.shape[1]):
            # print(preds[:, i, :].squeeze().shape, labels[:, i].shape)
            losses += self.loss(preds[:, i, :].squeeze(), labels[:, i])
        # losses = losses/preds.shape[1]
        return losses

class ARCNN_accuracy(nn.Module):
    def __init__(self):
        super().__init__()
        self.accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=2).to(DEVICE)

    def get_labels(self, preds, target):
        # print(target.shape)
        # labels = target.unsqueeze(-1)
        labels = target.expand(preds.size()[0], preds.size()[1])
        # print(labels.shape)
        return labels
    
    def __call__(self, preds, target_is_real):
        accuracies = 0
        labels = self.get_labels(preds, target_is_real)
    
        for i in range(preds.shape[1]):
            # print(preds[:, i, :].squeeze().shape, labels[:, i].shape)
            accuracies += self.accuracy(preds[:, i, :].squeeze(), labels[:, i])
        accuracies = accuracies/preds.shape[1]
        return accuracies

class ACRNN(pl.LightningModule):
    def __init__(self, n_classes, eeg_channels, cwa_hidden, cnn_in_channels, rnn_hidden, rnn_embed_dim, dropout = 0.0):
        super().__init__()

        # model architecture
        self.cwa = ChannelWiseAttention(n_channels=eeg_channels, hidden_size=cwa_hidden)
        self.cnn = CNN(cnn_in_channels)
        self.rnn_attn = RNN_Attention(hidden_size=rnn_hidden, embed_dim=rnn_embed_dim, dropout=dropout)
        # 加入卷积层以实现特征融合
        # self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.linear_out_valence = nn.Linear(rnn_hidden * 2, n_classes)
        self.linear_out_arousal = nn.Linear(rnn_hidden * 2, n_classes)
        self.linear_out_like = nn.Linear(rnn_hidden * 2, n_classes)
        # training
        # self.train_set, self.val_set = dataset_prepare(n_subjects=N_SUBJECT)
        # self.accuracy = pl.metrics.Accuracy()
        self.accuracy = ARCNN_accuracy()

        self.crossEntropyLoss_valence = ARCNN_loss()
        self.crossEntropyLoss_arousal = ARCNN_loss()
        self.crossEntropyLoss_like = ARCNN_loss()

    def forward(self, x):
        # input torch.Size([256, 3, 32, 128])
        x = self.cwa(x)
        # after cwa torch.Size([256, 3, 32, 128])
        x = self.cnn(x)
        # after cnn torch.Size([256, 3, 80])
        x = self.rnn_attn(x)
        # after rnn_attn torch.Size([256, 3, 128])

        output_valence = self.linear_out_valence(x)
        # output  torch.Size([256, 3, 2])
         
        output_arousal = self.linear_out_arousal(x)
        output_like = self.linear_out_like(x)
        return output_valence, output_arousal, output_like

    
    # def train_dataloader(self) -> DataLoader:
    #     return DataLoader(dataset = self.train_set, batch_size = BATCH_SIZE, shuffle = True)

    # def val_dataloader(self) -> DataLoader:
    #     return DataLoader(dataset = self.val_set, batch_size = BATCH_SIZE)

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=LR)
        return torch.optim.Adam(self.parameters(), lr = LR)

    def training_step(self, batch, _batch_idx): # pylint: disable=arguments-differ
        inputs, (labels_valence, labels_arousal, labels_like) = batch
        labels_valence = labels_valence.type(torch.LongTensor).to(DEVICE)
        labels_arousal = labels_arousal.type(torch.LongTensor).to(DEVICE)
        labels_like = labels_like.type(torch.LongTensor).to(DEVICE)

        output_valence, output_arousal, output_like = self(inputs)

        loss_valence = self.crossEntropyLoss_valence(output_valence, labels_valence)
        loss_arousal = self.crossEntropyLoss_arousal(output_arousal, labels_arousal)
        loss_like = self.crossEntropyLoss_like(output_like, labels_like)

        loss = 0.4 * loss_valence + 0.4 * loss_arousal + 0.2 * loss_like
        self.log('train_loss', loss)

        accuracy = self.accuracy(output_valence, labels_valence)
        self.log('train_acc', accuracy)
        
        return loss

    def evaluate(self, batch, stage=None):
        inputs, (labels_valence, labels_arousal, labels_like) = batch
        labels_valence = labels_valence.type(torch.LongTensor).to(DEVICE)
        labels_arousal = labels_arousal.type(torch.LongTensor).to(DEVICE)
        labels_like = labels_like.type(torch.LongTensor).to(DEVICE)

        output_valence, output_arousal, output_like = self(inputs)
        
        loss_valence = self.crossEntropyLoss_valence(output_valence, labels_valence)
        loss_arousal = self.crossEntropyLoss_arousal(output_arousal, labels_arousal)
        loss_like = self.crossEntropyLoss_like(output_like, labels_like)

        loss = 0.4 * loss_valence + 0.4 * loss_arousal + 0.2 * loss_like

        accuracy = self.accuracy(output_valence, labels_valence)

        if stage:
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_acc', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

class MultiTaskDataset(Dataset):
    def __init__(self, eeg_data, valence_labels, arousal_labels, like_labels):
        self.eeg_data = eeg_data
        self.valence_labels = valence_labels
        self.arousal_labels = arousal_labels
        self.like_labels = like_labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_sample = self.eeg_data[idx]
        valence_label = self.valence_labels[idx]
        arousal_label = self.arousal_labels[idx]
        like_label = self.like_labels[idx]
        return eeg_sample, (valence_label, arousal_label, like_label)
    
if __name__ == "__main__":
    model = ACRNN(
        n_classes = 2, 
        eeg_channels = 32,
        cwa_hidden = 64, 
        cnn_in_channels = 1, 
        rnn_hidden = 64, 
        rnn_embed_dim = 128,
        dropout = 0.2
    ).to(DEVICE)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        save_weights_only=True,
        verbose=True,
        dirpath=f'weights/ACRNN_sub_all_subjects',
        filename= f"ACRNN"+"-{epoch:02d}-{val_loss:.2f}.ckpt"
    )
    tb_logger = pl_loggers.TensorBoardLogger(name = f"ACRNN_sub_all_subjects", save_dir = 'lightning_logs')

    trainer = pl.Trainer(
        gpus = NUM_GPUS,
        max_epochs = EPOCHS,
        # accumulate_grad_batches = 1,
        # auto_lr_find=True,
        callbacks = [checkpoint_callback],
        logger=tb_logger,
        # val_check_interval=0.25,
        check_val_every_n_epoch = 1,
        precision=16,
        # resume_from_checkpoint="",
        # default_save_path = './weights'
    )

    # all subjects shuffle then split
    # train_set, val_set = dataset_prepare(segment_duration = 3, n_subjects = 1, load_all = True, single_subject = False, sampling_rate = 128)
    
    # # # all subjects leave some subjects out
    X_train_val, X_test_val, y_train_val, y_test_val, X_train_arousal, X_test_arousal, y_train_arousal, y_test_arousal, X_train_like, X_test_like, y_train_like, y_test_like = dataset_prepare(segment_duration = 3, n_subjects = 26, load_all = True, single_subject = False, sampling_rate = 128)

    train_dataset = MultiTaskDataset(X_train_val, y_train_val, y_train_arousal, y_train_like)
    test_dataset = MultiTaskDataset(X_test_val, y_test_val, y_test_arousal, y_test_like)

    train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True) 
    val_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    trainer.fit(model, train_loader, val_loader)

    
    # # perform k-fold:
    # for nth_subject in range(32):
    #     X, y = dataset_prepare_for_KF(n_subjects=nth_subject)
    #     kf = KFold(10, shuffle = True, random_state = 29)

    #     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    #         print("training fold: ", fold+1)
    #         X_train, X_val = X[train_idx], X[val_idx]
    #         y_train, y_val = y[train_idx], y[val_idx]

    #         train_x = torch.Tensor(X_train) # transform to torch tensor
    #         train_y = torch.Tensor(y_train)
    #         val_x = torch.Tensor(X_val) # transform to torch tensor
    #         val_y = torch.Tensor(y_val)

    #         train_set = TensorDataset(train_x, train_y.long()) # create your datset
    #         val_set = TensorDataset(val_x, val_y.long())
    #         train_loader = DataLoader(dataset = train_set, batch_size = BATCH_SIZE, shuffle = True)
    #         val_loader = DataLoader(dataset = val_set, batch_size = BATCH_SIZE, shuffle = False)

    #         model = ACRNN(
    #         n_classes = 2, 
    #         eeg_channels = 32,
    #         cwa_hidden = 64, 
    #         cnn_in_channels = 1, 
    #         rnn_hidden = 64, 
    #         rnn_embed_dim = 128,
    #         ).to(DEVICE)

    #         checkpoint_callback = ModelCheckpoint(
    #             monitor='val_loss',
    #             save_weights_only=True,
    #             verbose=True,
    #             dirpath=f'weights/subject_{nth_subject}/ACRNN_fold_{fold+1}',
    #             filename= f"ACRNN"+"-{epoch:02d}-{val_loss:.2f}.ckpt"
    #         )
    #         tb_logger = pl_loggers.TensorBoardLogger(name = f"ACRNN_fold_{fold+1}", save_dir = f'lightning_logs/subject_{nth_subject}')

    #         trainer = pl.Trainer(
    #             gpus = NUM_GPUS,
    #             max_epochs = EPOCHS,
    #             # accumulate_grad_batches = 1,
    #             # auto_lr_find=True,
    #             callbacks = [checkpoint_callback],
    #             logger=tb_logger,
    #             # val_check_interval=0.25,
    #             check_val_every_n_epoch = 1,
    #             precision=16,
    #             # resume_from_checkpoint="",
    #             # default_save_path = './weights'
    #         )

    #         trainer.fit(model, train_loader, val_loader)
