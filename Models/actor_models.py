import time
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Models.embedding_models import ConvNet
from Models.attention_module import MultiHeadAttention

class ConvActor(nn.Module):
    def __init__(self, input_channels, output_dim=3, max_action=5, with_SE=False, vis=False, activation='softmax'):
        super(ConvActor, self).__init__()
        self.embedding = ConvNet(input_channels, with_SE, vis=vis)

        self.fc1 = nn.Linear(128*5*5, 512)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, output_dim)

        assert activation in ['Sigmoid', 'Tanh', 'Softmax', 'Relu', 'No'], \
            print('Unsupported output activation function!')
        self.activation = activation
        self._init_weights()
        self.vis = vis
        self.max_action = max_action
        self.attention_visualization = {}

    def forward(self, image, vector=None, hidden=None):
        self.attention_visualization.update({'input': image})
        feature = self.embedding(image)
        self.attention_visualization.update({'embedding_feature': feature})
        feature = torch.flatten(feature, 1)
        out = F.relu(self.fc1(feature))
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        if self.activation == 'Sigmoid':
            out = F.sigmoid(self.fc3(out))
        elif self.activation == 'Tanh':
            out = torch.tanh(self.fc3(out))
        elif self.activation == 'Relu':
            out = F.relu(self.fc3(out))
        elif self.activation == 'Softmax':
            out = F.softmax(self.fc3(out))
        else:
            out = self.fc3(out)
        return out * self.max_action

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d or nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def visualization_clear(self):
        self.attention_visualization.clear()
        self.embedding.attention_visualization.clear()

    def visualization(self, vis_dir='', fmt='pdf'):
        vis_time = time.time()
        # visualize overall network
        if len(self.attention_visualization) > 0:
            for k, tensor in self.attention_visualization.items():
                sample = tensor[0]
                # sample: CxHxW
                sample = sample.permute(1, 2, 0)  # HxWxC
                file_name = os.path.join(vis_dir, '{}_{}.{}'.format(vis_time, k, fmt))
                if k == 'input':
                    plt.imshow(sample[..., :3].cpu().numpy().astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
                else:
                    attention_map = sample.pow(2).mean(2).detach().cpu().numpy()  # HxWx1
                    attention_map = cv2.resize(src=attention_map, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
                    plt.imshow(attention_map.astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
        self.embedding.visualization(vis_dir=vis_dir, vis_time=vis_time)


class DiscreteActorHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DiscreteActorHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(),
        )

    def forward(self, x):
        return self.fc(x)


class ContinueActorHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, max_action=2):
        super(ContinueActorHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.max_action = max_action
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.sigma = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc(x)
        mu = torch.tanh(self.mu(out)) * self.max_action
        sigma = F.softplus(self.sigma(out))
        return mu, sigma


class Conv_LSTM_Actor(nn.Module):
    def __init__(
            self, input_channels, output_dim=3, max_action=5,
            lstm_dim=512, lstm_layers=2,
            with_SE=False, vis=False, activation='Softmax'):
        super(Conv_LSTM_Actor, self).__init__()
        self.embedding = ConvNet(input_channels, with_SE, vis=vis)
        self.fc1 = nn.Linear(128*5*5, 512)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 64)
        self.fc3 = nn.Linear(64, output_dim)

        assert activation in ['Sigmoid', 'Tanh', 'Softmax', 'Relu', 'No'], \
            print('Unsupported output activation function!')
        self.activation = activation
        self._init_weights()
        self.vis = vis
        self.max_action = max_action
        self.attention_visualization = {}

    def forward(self, image, vector=None, hidden=None):
        self.attention_visualization.update({'input': image})
        feature = self.embedding(image)
        self.attention_visualization.update({'embedding_feature': feature})
        feature = torch.flatten(feature, 1).unsqueeze(0)
        feature = F.relu(self.fc1(feature))
        out, hidden = self.lstm(feature, hidden)
        out = out.squeeze(0)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        if self.activation == 'Sigmoid':
            out = F.sigmoid(self.fc3(out))
        elif self.activation == 'Tanh':
            out = torch.tanh(self.fc3(out))
        elif self.activation == 'Relu':
            out = F.relu(self.fc3(out))
        elif self.activation == 'Softmax':
            out = F.softmax(self.fc3(out))
        else:
            out = self.fc3(out)
        return out * self.max_action, hidden

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d or nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def visualization_clear(self):
        self.attention_visualization.clear()
        self.embedding.attention_visualization.clear()

    def visualization(self, vis_dir='', fmt='pdf'):
        vis_time = time.time()
        # visualize overall network
        if len(self.attention_visualization) > 0:
            for k, tensor in self.attention_visualization.items():
                sample = tensor[0]
                # sample: CxHxW
                sample = sample.permute(1, 2, 0)  # HxWxC
                file_name = os.path.join(vis_dir, '{}_{}.{}'.format(vis_time, k, fmt))
                if k == 'input':
                    plt.imshow(sample[..., :3].cpu().numpy().astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
                else:
                    attention_map = sample.pow(2).mean(2).detach().cpu().numpy()  # HxWx1
                    attention_map = cv2.resize(src=attention_map, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
                    plt.imshow(attention_map.astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
        self.embedding.visualization(vis_dir=vis_dir, vis_time=vis_time)


class Conv_LSTM_MHA_Actor(nn.Module):
    def __init__(
            self, input_channels, output_dim=3, max_action=5,
            lstm_dim=512, lstm_layers=2,
            attention_type='MHA', head_num=8,
            with_SE=False, vis=False, activation='softmax'):
        super(Conv_LSTM_MHA_Actor, self).__init__()
        self.embedding = ConvNet(input_channels, with_SE, vis=vis)
        self.attention_type = attention_type
        self.attention = MultiHeadAttention(128, head_num)
        self.fc1 = nn.Linear(128*5*5, 512)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 64)
        self.fc3 = nn.Linear(64, output_dim)

        assert activation in ['Sigmoid', 'Tanh', 'Softmax', 'Relu', 'No'], \
            print('Unsupported output activation function!')
        self.activation = activation
        self._init_weights()
        self.vis = vis
        self.max_action = max_action
        self.attention_visualization = {}

    def forward(self, image, vector=None, hidden=None):
        self.attention_visualization.update({'input': image})
        feature = self.embedding(image)
        self.attention_visualization.update({'embedding_feature': feature})
        attention_map = self.attention(feature).transpose(-1, -2).view(*feature.shape)  # BxCxWxH
        self.attention_visualization.update({'attention_map': attention_map})
        context = feature * attention_map
        self.attention_visualization.update({'context': context})
        context = F.relu(self.fc1(torch.flatten(context, start_dim=1))).unsqueeze(0)

        out, hidden = self.lstm(context, hidden)
        out = out.squeeze(0)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        if self.activation == 'Sigmoid':
            out = F.sigmoid(self.fc3(out))
        elif self.activation == 'Tanh':
            out = torch.tanh(self.fc3(out))
        elif self.activation == 'Relu':
            out = F.relu(self.fc3(out))
        elif self.activation == 'Softmax':
            out = F.softmax(self.fc3(out))
        else:
            out = self.fc3(out)
        return out * self.max_action, hidden

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d or nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

    def visualization_clear(self):
        self.attention_visualization.clear()
        self.embedding.attention_visualization.clear()

    def visualization(self, vis_dir='', fmt='pdf'):
        vis_time = time.time()
        # visualize overall network
        if len(self.attention_visualization) > 0:
            for k, tensor in self.attention_visualization.items():
                sample = tensor[0]
                # sample: CxHxW
                sample = sample.permute(1, 2, 0)  # HxWxC
                file_name = os.path.join(vis_dir, '{}_{}.{}'.format(vis_time, k, fmt))
                if k == 'input':
                    plt.imshow(sample[..., :3].cpu().numpy().astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
                else:
                    attention_map = sample.pow(2).mean(2).detach().cpu().numpy()  # HxWx1
                    attention_map = cv2.resize(src=attention_map, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
                    plt.imshow(attention_map.astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
        self.embedding.visualization(vis_dir=vis_dir, vis_time=vis_time)