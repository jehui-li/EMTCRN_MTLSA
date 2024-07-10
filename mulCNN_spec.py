from scipy.sparse.construct import random
import torch 
import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from einops import rearrange, repeat
from torch.utils.data import DataLoader, TensorDataset, Dataset

from preprocessing import dataset_prepare
from sklearn.model_selection import KFold
import torchaudio.transforms as T
from torchmetrics.classification import F1Score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
EPOCHS = 26
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

        s = torch.mean(x, dim = -1)
        # s_shape = (batch_size, n_steps, n_channel)
        # after torch.mean ([256, 3, 32])

        s = self.linear_in(s)
        # s_shape = (batch_size, n_steps, hidden_size)
        # after linear_in: ([256, 3, 64])

        s = self.linear_out(s)
        # s_shape = (batch_size, n_steps, n_channel)
        # after linear_out: ([256, 3, 32])

        s = self.tanh(s)
        # after tanh: ([256, 3, 32])

        attention_map = self.softmax(s)
        # attention_map shape = (batch_size, n_steps, n_channel, 1)
        # after attention_map: torch.Size([256, 3, 32])

        attention_map = attention_map.unsqueeze(-1)
        # after unsqueeze: torch.Size([256, 3, 32, 1])

        # calculate output: cj = aj * xj
        out = x * attention_map
        # out: torch.Size([256, 3, 32, 128])

        # [256, 3, 32, 128]
        return out

class CNN1(nn.Module):
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
            out_channels = 40, kernel_size = 11)  # 输入通道数为特征向量的维度
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
        # after conv [768, 40, 80]

        x = x.unsqueeze(2)
        # after unsqueeze ([768, 40, 1, 80])
        
        x = self.elu(x)
        # after elu ([768, 40, 1, 80])

        x = self.pool(x)
        print("pool1", x.shape)
        # after pool ([768, 40, 1, 1])

        x = self.dropout(x)
        # dropout ([768, 40, 1, 1])

        out = x.view(batch, step, -1) 
        # out ([256, 3, 40])
        return out

class CNN2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels = in_channels, # in_channels = 1
            out_channels = 5, # 卷积核的数量
            kernel_size = (8, 65), 
            stride = (4, 1)
        )
        self.elu = nn.ELU()
        self.conv1d = nn.Conv1d(
            in_channels = 5, 
            out_channels = 5, kernel_size = 49)  # 输入通道数为特征向量的维度
        self.pool = nn.MaxPool2d(kernel_size = (1, 120), stride = 40)
        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        # input shape: (batch_size, n_steps, n_channels, n_samples)
        # [256, 3, 32, 128]

        batch, step, h, w = x.shape
        x = x.view(batch * step, 1, h, w) 
        # reshape to do parallel convolution on every step
        # new_shape = (batch * n_step, 1, n_channels, n_samples)
        # [768, 1, 32, 128]

# conv2d torch.Size([768, 5, 7, 64])
# view torch.Size([768, 20, 112])
# conv1d torch.Size([768, 20, 50])
# pool2 torch.Size([768, 20, 1, 2])

        x = self.conv2d(x)
        print("conv2d", x.shape)
        # after conv [768, 5, 7, 64]

        x = self.elu(x)
        # after elu ([768, 5, 7, 64])

        x = x.view(768, 5, -1)
        print("view", x.shape)
        # after squeeze ([768, 5, 448])

        x = self.conv1d(x)
        print("conv1d", x.shape)
        # after conv [768, 5, 400]

        x = x.unsqueeze(2)
        # after unsqueeze ([768, 5, 1, 400])
        
        x = self.elu(x)
        # after elu ([768, 5, 1, 400])

        x = self.pool(x)
        print("pool2", x.shape)
        # after pool ([768, 5, 1, 8])

        x = self.dropout(x)
        # dropout ([768, 5, 1, 8])

        out = x.view(batch, step, -1) 
        # out ([256, 3, 40])
        return out

class ChannelFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn1 = CNN1(in_channels)
        self.cnn2 = CNN2(in_channels)

    def forward(self, x):
        out1 = self.cnn1(x)
        out2 = self.cnn2(x)
        # Concatenate the outputs along the last dimension
        out = torch.cat((out1, out2), dim=2)
        # ChannelFusion out ([256, 3, 80])
        return out

class SpectrogramLayer(nn.Module):
    def __init__(self, n_fft=64, hop_length=32, sampling_rate=128):
        super(SpectrogramLayer, self).__init__()
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2, pad=64)
        self.batch_norm = nn.BatchNorm1d(21)  # 对应于通道数

    def forward(self, x):
        # print("x", x.shape) x[256, 3, 80] (batch_size, n_steps, n_samples)
        # 计算频谱图
        batch_size, time_steps, samples = x.shape
        x = self.spectrogram(x)  # 计算频谱图
        # print("spectrogram", x.shape)# [256, 3, 129, 7] [batch_size, n_steps, n_fft/2, n_samples/hop_length + 1]
        # [256, 3, 129, 7](batch_size, time_steps, freq_bins频率分辨率, time_steps)
        
        # 调整频谱图维度以适应批量归一化
        new_shape = (x.size(0), -1, x.size(2)) 
        x = x.reshape(*new_shape) # [256, 21, 129] (batch_size, time_steps, freq_bins频率分辨率)

        # 批量归一化
        x = self.batch_norm(x) # (batch_size, time_steps, freq_bins频率分辨率)
        # print("x", x.shape) 
        reshaped = x.view(x.size(0), 3, -1, x.size(2)) # [256, 3, 7, 129]

        output = reshaped.view(x.size(0), 3, -1) # 然后将最后两个维度展平为[256, 3, 903]
        return output

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

class Bi_LSTM_Attention(nn.Module):
    def __init__(self, hidden_size = 64, embed_dim = 128, num_layers = 2, num_heads = 1, dropout = 0.2, sparsity=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.biLstm = nn.LSTM(
            input_size = 80, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.self_attention = Attention(
            dim=hidden_size * 2, 
            heads=num_heads,
            dim_head=embed_dim,
            dropout=dropout,
            sparsity=sparsity,
        )

    def forward(self, x):
        # initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) 
        # initialize cell state for LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device) 
        # lstm input torch.Size([256, 3, 40])
        r_out, _ = self.biLstm(x, (h0, c0))
        # lstm output torch.Size([256, 3, 128])
        output = self.self_attention(r_out)
        # lstm_attn output torch.Size([256, 3, 128])
        return output

class Multi_ARCNN_loss(nn.Module):
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

class Multi_ARCNN_accuracy(nn.Module):
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

class Multi_ACRNN(pl.LightningModule):
    def __init__(self, n_classes, eeg_channels, cwa_hidden, cnn_in_channels, rnn_hidden, rnn_embed_dim, dropout, sampling_rate):
        super().__init__()
        # [256, 3, 32, 128]
        self.cwa = ChannelWiseAttention(n_channels=eeg_channels, hidden_size=cwa_hidden)
        # [256, 3, 32, 128]
        self.cnn = ChannelFusion(cnn_in_channels)
        # cnn ([256, 3, 80])
        self.spec_layer = SpectrogramLayer(sampling_rate)
        # spec_layer
        self.rnn_attn = Bi_LSTM_Attention(hidden_size=rnn_hidden, 
                    embed_dim=rnn_embed_dim, dropout=dropout)
        self.linear_out_valence = nn.Linear(1031, n_classes)
        self.linear_out_arousal = nn.Linear(1031, n_classes)

        self.accuracy = Multi_ARCNN_accuracy()

        self.crossEntropyLoss_valence = Multi_ARCNN_loss()
        self.crossEntropyLoss_arousal = Multi_ARCNN_loss()

        # self.f1 = F1Score(num_classes = n_classes, average='macro', task='multiclass')

    def forward(self, x):
        # input torch.Size([256, 3, 32, 128])
        x = x.to(DEVICE)
        x = self.cwa(x)
        # after cwa torch.Size([256, 3, 32, 128])
        x = self.cnn(x).to(DEVICE)
        # print("cnn", x.shape)
        # after cnn torch.Size([256, 3, 80])
        x1 = self.spec_layer(x).to(DEVICE)
        # print("spec_layer", x1.shape)  [256, 3, 903]
        x2 = self.rnn_attn(x).to(DEVICE)
        # after rnn_attn torch.Size([256, 3, 128])
        x = torch.cat((x1, x2), dim=2)
        # print("cat", x.shape) [256, 3, 1031]
        output_valence = self.linear_out_valence(x)
        # output  torch.Size([256, 3, 2])
         
        output_arousal = self.linear_out_arousal(x)
        return output_valence, output_arousal

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=LR)
        return torch.optim.Adam(self.parameters(), lr = LR)

    def training_step(self, batch, _batch_idx): # pylint: disable=arguments-differ
        inputs, (labels_valence, labels_arousal) = batch
        inputs = inputs.to(DEVICE)
        labels_valence = labels_valence.type(torch.LongTensor).to(DEVICE)
        labels_arousal = labels_arousal.type(torch.LongTensor).to(DEVICE)
        output_valence, output_arousal = self(inputs)

        loss_valence = self.crossEntropyLoss_valence(output_valence, labels_valence)
        loss_arousal = self.crossEntropyLoss_arousal(output_arousal, labels_arousal)
        loss = 0.75 * loss_valence + 0.25 * loss_arousal
        self.log('train_loss', loss)

        accuracy = self.accuracy(output_valence, labels_valence)
        self.log('train_acc', accuracy)
        

        return loss

    def evaluate(self, batch, stage=None):
        inputs, (labels_valence, labels_arousal) = batch
        inputs = inputs.to(DEVICE)
        labels_valence = labels_valence.type(torch.LongTensor).to(DEVICE)
        labels_arousal = labels_arousal.type(torch.LongTensor).to(DEVICE)
        output_valence, output_arousal = self(inputs)
        
        loss_valence = self.crossEntropyLoss_valence(output_valence, labels_valence)
        loss_arousal = self.crossEntropyLoss_arousal(output_arousal, labels_arousal)
        loss = 0.5 * loss_valence + 0.5 * loss_arousal

        accuracy = self.accuracy(output_valence, labels_valence)
        preds = torch.argmax(output_valence, dim=1)
        # f1 = self.f1(preds, labels_valence)

        if stage:
            self.log(f'{stage}_loss', loss)
            self.log(f'{stage}_acc', accuracy)
            # self.log(f'{stage}_F1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, 'test')

class MultiTaskDataset(Dataset):
    def __init__(self, eeg_data, valence_labels, arousal_labels):
        self.eeg_data = eeg_data
        self.valence_labels = valence_labels
        self.arousal_labels = arousal_labels

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg_sample = self.eeg_data[idx]
        valence_label = self.valence_labels[idx]
        arousal_label = self.arousal_labels[idx]
        return eeg_sample, (valence_label, arousal_label)
    
if __name__ == "__main__":
    model = Multi_ACRNN(
        n_classes = 2, 
        eeg_channels = 32,
        cwa_hidden = 64, 
        cnn_in_channels = 1, 
        rnn_hidden = 64, 
        rnn_embed_dim = 128,
        dropout = 0.2,
        sampling_rate = 256
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
    X_train_val, X_test_val, X_value_val, y_train_val, y_test_val, y_value_val, X_train_arousal, X_test_arousal, X_val_arousal, y_train_arousal, y_test_arousal, y_val_arousal = dataset_prepare(segment_duration = 3, n_subjects = 26, load_all = True, single_subject = False, sampling_rate = 128)

    train_dataset = MultiTaskDataset(X_train_val, y_train_val, y_train_arousal)
    test_dataset = MultiTaskDataset(X_test_val, y_test_val, y_test_arousal)
    value_dataset = MultiTaskDataset(X_value_val, y_value_val, y_val_arousal)

    train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True) 
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)
    val_loader = DataLoader(dataset = value_dataset, batch_size = BATCH_SIZE, shuffle = False) 

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)