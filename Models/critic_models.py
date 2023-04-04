import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import cv2
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Models.attention_module import AddictiveAttention, DotProdAttention, MultiHeadAttention
from Models.function_modules import SELayer
from torch.autograd import Variable
from Models.embedding_models import ConvNet, ResNet


class CriticHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(CriticHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.fc(x)


class FCNet(nn.Module):
    def __init__(self, vec_dim=6, output_dim=64, history_len=1):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(vec_dim, 32)
        self.fc2 = nn.Linear(32 * history_len, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self._init_weights()

    def forward(self, image=None, vector=None, hidden=None):
        feature = F.relu(self.fc1(vector))  # BxMxN or BxN (M is the history length)
        feature = torch.flatten(feature, start_dim=1)  # BxN
        out = F.relu(self.fc2(feature))
        out = self.fc3(out)
        return out

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
        return

    def visualization(self, vis_dir='', fmt='pdf'):
        return


class FC_LSTM(nn.Module):
    def __init__(self, vec_dim=6, action_dim=11, lstm_dim=128, lstm_layers=2, history_len=1):
        super(FC_LSTM, self).__init__()
        self.name = 'FC_LSTM'
        self.embedding = FCNet(vec_dim, lstm_dim, history_len)

        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 32)
        self.fc3 = nn.Linear(32, action_dim)

    def forward(self, image, vector, hidden=None):
        # input1 dim: BxN
        feature = F.relu(self.embedding(vector=vector)).unsqueeze(0)  # 1xBx128
        out, hidden = self.lstm(feature, hidden)
        out = out.squeeze(0)  # Bx128
        out = self.dropout(out)
        out = F.relu(self.fc2(out))  # Bx64
        out = self.fc3(out)
        return out, hidden

    def visualization_clear(self):
        return

    def visualization(self, vis_dir='', fmt='pdf'):
        return


class Conv_LSTM_QNet(nn.Module):
    def __init__(
            self, image_channels, action_dim=None,
            lstm_dim=1024, lstm_layers=2,
            vis_dir='', with_SE=True, vis=False):
        super(Conv_LSTM_QNet, self).__init__()
        self.name = 'Conv_LSTM_QNet'
        self.embedding = ConvNet(image_channels, with_SE, vis=vis)
        if action_dim is None:
            self.fc1 = nn.Linear(128 * 5 * 5, lstm_dim)
        else:
            self.fc1 = nn.Linear(128 * 5 * 5 + action_dim, lstm_dim)

        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 256)
        self.fc3 = nn.Linear(256, 1)

        self._init_weights()
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, image, vector, action=None, hidden=None):
        # state dim: BxCxWxH
        self.attention_visualization.update({'input': image})
        feature = self.embedding(image)
        self.attention_visualization.update({'embedding_feature': feature})
        feature = torch.flatten(feature, 1)
        if action is not None:
            action = torch.flatten(action, 1)
            feature = torch.cat([feature, action], dim=1)
        feature = F.relu(self.fc1(feature)).unsqueeze(0)
        out, hidden = self.lstm(feature, hidden)
        out = out.squeeze(0)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out, hidden

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


class Conv_LSTM_MHA_QNet(nn.Module):
    def __init__(
            self, image_channels, action_dim=None,
            lstm_dim=1024, lstm_layers=2,
            attention_type='MHA', head_num=4,
            with_SE=True, vis=False, vis_dir=''):
        super(Conv_LSTM_MHA_QNet, self).__init__()
        self.name = 'Conv_LSTM_MHA_QNet'
        self.lstm_dim = lstm_dim
        self.lstm_layer = lstm_layers
        self.embedding = ConvNet(image_channels, with_SE, vis=vis)
        self.attention_type = attention_type
        self.attention = MultiHeadAttention(128, head_num)
        if action_dim is None:
            self.fc1 = nn.Linear(128 * 5 * 5, lstm_dim)
        else:
            self.fc1 = nn.Linear(128 * 5 * 5 + action_dim, lstm_dim)

        # self.lstm = nn.LSTM(128, lstm_dim, lstm_layers)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 256)
        self.fc3 = nn.Linear(256, 1)

        self._init_weights()
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, input1, input2, action=None, hidden=None):
        self.attention_visualization.update({'input': input1})
        # state dim: BxCxWxH
        feature = self.embedding(input1)  # BxCxWxH
        self.attention_visualization.update({'embedding_feature': feature})
        attention_map = self.attention(feature).transpose(-1, -2).view(*feature.shape)  # BxCxWxH
        self.attention_visualization.update({'attention_map': attention_map})
        context = feature * attention_map
        self.attention_visualization.update({'context': context})
        # context = F.layer_norm(feature + res_context, normalized_shape=res_context.shape[1:])  # BxCxWxH
        context = torch.flatten(context, 1)
        if action is not None:
            action = torch.flatten(action, 1)
            context = torch.cat([context, action], dim=1)
        context = F.relu(self.fc1(context)).unsqueeze(0)  # 1xBxC

        # context = self.attention(feature) # Bx(W*H)xC
        # context = torch.sum(context, dim=1).unsqueeze(0)  # 1xBxC
        out, hidden = self.lstm(context, hidden)  # 1xBxC
        out = out.squeeze(0)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out, hidden

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
                print('=> saving {} attention map'.format(k))
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

        self.embedding.visualization(vis_dir, vis_time)


class Conv_LSTM(nn.Module):
    def __init__(
            self, image_channels, action_dim=11,
            lstm_dim=1024, lstm_layers=2,
            vis_dir='', with_SE=True, vis=False):
        super(Conv_LSTM, self).__init__()
        self.name = 'Conv_LSTM'
        self.embedding = ConvNet(image_channels, with_SE, vis=vis)

        self.fc1 = nn.Linear(128 * 5 * 5, lstm_dim)

        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        # self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self._init_weights()
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, image, vector, hidden=None):
        # state dim: BxCxWxH
        self.attention_visualization.update({'input': image})
        feature = self.embedding(image)
        self.attention_visualization.update({'embedding_feature': feature})
        feature = torch.flatten(feature, 1).unsqueeze(0)
        feature = F.relu(self.fc1(feature))
        out, hidden = self.lstm(feature, hidden)
        out = out.squeeze(0)
        # out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out, hidden

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


class Multimodal_LSTM(nn.Module):
    def __init__(self, image_channels, vec_dim=3, action_dim=11, lstm_dim=1024, lstm_layers=2,
                 history_len=1, vis_dir='', with_SE=True, vis=False):
        super(Multimodal_LSTM, self).__init__()
        self.name = 'Multi_Modal_LSTM'
        self.embedding = ConvNet(image_channels, with_SE, vis)
        self.fc1 = nn.Linear(128 * 5 * 5, lstm_dim - 64)
        self.vec_embedding = FCNet(vec_dim, output_dim=64, history_len=history_len)

        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self._init_weights()
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, image, vector, hidden=None):
        # input1 dim: BxCxWxH
        # input2 dim: BxN
        self.attention_visualization.update({'input': image})
        image_feature = self.embedding(image)  # Bx128x5x5
        self.attention_visualization.update({'embedding_feature': image_feature})
        vec_feature = F.relu(self.vec_embedding(vector=vector)).unsqueeze(0)  # 1xBx128
        image_feature = torch.flatten(image_feature, 1).unsqueeze(0)  # 1xBx3200
        image_feature = F.relu(self.fc1(image_feature))  # 1xBxL
        feature = torch.cat([image_feature, vec_feature], dim=-1)  # 1xBx(L+128)
        out, hidden = self.lstm(feature, hidden)
        out = out.squeeze(0)  # Bx(L+128)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))  # Bx256
        out = self.fc3(out)
        return out, hidden

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


class Conv_LSTM_A(nn.Module):
    def __init__(self, image_channels, action_dim=11, lstm_dim=1024, lstm_layers=2, attention_type='Add',
                 with_SE=True, vis=False, vis_dir=''):
        super(Conv_LSTM_A, self).__init__()
        self.name = 'Conv_LSTM_Attention'
        self.lstm_dim = lstm_dim
        self.lstm_layer = lstm_layers
        self.embedding = ConvNet(image_channels, with_SE, vis=vis)
        self.attention_type = attention_type
        if self.attention_type == 'Add':
            self.attention = AddictiveAttention(128, lstm_dim // 8, lstm_dim)
        elif self.attention_type == 'DotProd':
            self.attention = DotProdAttention(128, lstm_dim // 8, lstm_dim)
        else:
            raise ValueError('Unsupported attention type!')

        self.lstm = nn.LSTM(128, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self._init_weights()
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, image, vector, hidden=None):
        # state dim: BxCxWxH
        self.attention_visualization.update({'input': image})
        feature = self.embedding(image)  # BxCxWxH
        self.attention_visualization.update({'embedding_feature': feature})
        if hidden is None:
            h = torch.zeros(self.lstm_layer, image.shape[0], self.lstm_dim)
            attention, context = self.attention(feature, h[-1])  # Bx(W*H), BxC
        else:
            attention, context = self.attention(feature, torch.sum(hidden[0], dim=0) / self.lstm_layer)  # Bx(W*H), BxC
        self.attention_visualization.update({'attention': attention.view(-1, *feature.shape[-2:]).unsqueeze(0)})

        out, hidden = self.lstm(context.unsqueeze(0), hidden)  # 1xBxN
        out = out.squeeze(0)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        if not self.vis:
            self.attention_visualization.clear()

        return out, hidden

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
                print('=> saving {} attention map'.format(k))
                sample = tensor[0]
                # sample: CxHxW
                sample = sample.permute(1, 2, 0)  # HxWxC
                file_name = os.path.join(vis_dir, '{}_{}.{}}'.format(vis_time, k, fmt))
                if k == 'input':
                    plt.imshow(sample[..., :3].cpu().numpy().astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
                else:
                    attention_map = sample.pow(2).mean(2).detach().cpu().numpy()  # HxW
                    attention_map = cv2.resize(src=attention_map, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
                    plt.imshow(attention_map.astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
        self.embedding.visualization(vis_dir=vis_dir, vis_time=vis_time)


class Conv_LSTM_MHA(nn.Module):
    def __init__(
            self, image_channels, action_dim=11,
            lstm_dim=1024, lstm_layers=2,
            attention_type='MHA', head_num=4,
            with_SE=True, vis=False, vis_dir=''):
        super(Conv_LSTM_MHA, self).__init__()
        self.name = 'Conv_LSTM_MHA'
        self.lstm_dim = lstm_dim
        self.lstm_layer = lstm_layers
        self.embedding = ConvNet(image_channels, with_SE, vis=vis)
        self.attention_type = attention_type
        self.attention = MultiHeadAttention(128, head_num)

        self.fc1 = nn.Linear(128 * 5 * 5, lstm_dim)

        # self.lstm = nn.LSTM(128, lstm_dim, lstm_layers)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(lstm_dim, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self._init_weights()
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, input1, input2, hidden=None):
        self.attention_visualization.update({'input': input1})
        # state dim: BxCxWxH
        feature = self.embedding(input1)  # BxCxWxH
        self.attention_visualization.update({'embedding_feature': feature})
        attention_map = self.attention(feature).transpose(-1, -2).view(*feature.shape)  # BxCxWxH
        self.attention_visualization.update({'attention_map': attention_map})
        context = feature * attention_map
        self.attention_visualization.update({'context': context})
        # context = F.layer_norm(feature + res_context, normalized_shape=res_context.shape[1:])  # BxCxWxH
        context = F.relu(self.fc1(torch.flatten(context, start_dim=1))).unsqueeze(0)  # 1xBxC

        # context = self.attention(feature) # Bx(W*H)xC
        # context = torch.sum(context, dim=1).unsqueeze(0)  # 1xBxC
        out, hidden = self.lstm(context, hidden)  # 1xBxC
        out = out.squeeze(0)
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out, hidden

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
                print('=> saving {} attention map'.format(k))
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

        self.embedding.visualization(vis_dir, vis_time)


class MultiModalQNet(nn.Module):
    def __init__(self, image_channels, vec_dim=3, action_dim=11, hidden_dim=256,
                 history_len=1, with_SE=True, vis=False):
        super(MultiModalQNet, self).__init__()
        self.name = 'Multi_Modal_QNet'
        self.embedding = ConvNet(image_channels, with_SE, vis)
        self.fc1 = nn.Linear(128 * 5 * 5, hidden_dim)
        self.vec_embedding = FCNet(vec_dim, output_dim=hidden_dim, history_len=history_len)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(2 * hidden_dim, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.vis = vis
        self.attention_visualization = {}

    def forward(self, image, vector=None, hidden=None):
        image_feature = self.embedding(image)
        image_feature = torch.flatten(image_feature, 1)
        image_feature = F.relu(self.fc1(image_feature))
        vec_feature = F.relu(self.vec_embedding(vector=vector))
        feature = torch.cat([image_feature, vec_feature], dim=1)
        feature = self.dropout(feature)
        out = F.relu(self.fc2(feature))
        out = self.fc3(out)
        return out

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


class ConvQNet(nn.Module):
    """docstring for Net"""

    def __init__(self, image_channels, output_dim=11, action_dim=None, with_SE=False, vis=False, vis_dir=''):
        super(ConvQNet, self).__init__()
        self.name = 'ConvQNet'
        self.embedding = ConvNet(image_channels, with_SE=with_SE, vis=vis)
        if action_dim is not None:
            self.fc1 = nn.Linear(128 * 5 * 5 + action_dim, 1024)
        else:
            self.fc1 = nn.Linear(128 * 5 * 5, 1024)

        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_dim)

        self._init_weights()
        self.vis = vis
        self.attention_visualization = {}

    def forward(self, image, vector=None, action=None):
        self.attention_visualization.update({'input': image})
        x = self.embedding(image)
        self.attention_visualization.update({'embedding_feature': x})
        x = torch.flatten(x, 1)
        if action is None:
            x = F.relu(self.fc1(x))
        else:
            x = torch.cat([x, action], dim=1)
            x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

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
                    plt.imshow(sample[..., :3].detach().cpu().numpy().astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')
                else:
                    attention_map = sample.pow(2).mean(2).detach().cpu().numpy()  # HxW
                    attention_map = cv2.resize(src=attention_map, dsize=(255, 255), interpolation=cv2.INTER_CUBIC)
                    plt.imshow(attention_map.astype(np.float))
                    # plt.tight_layout()
                    plt.savefig(file_name, bbox_inches='tight')

        self.embedding.visualization(vis_dir, vis_time)

        self.visualization_clear()


class ResQNet(nn.Module):
    def __init__(self, image_channels, output_dim=11, block=None, num_blocks=None):
        super(ResQNet, self).__init__()
        self.name = 'ResQNet'
        self.in_planes = 32

        self.conv1 = nn.Conv2d(image_channels, 32, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1)
        self.fc1 = nn.Linear(256 * block.expansion * 4 * 4, 2048)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, output_dim)

        self._init_weights()

    def _make_layer(self, block=None, planes=None, num_blocks=None, stride=2):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, image, vector):
        x = F.relu(self.bn1(self.conv1(image)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d or nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DuelingConvNet(nn.Module):
    def __init__(self, input_channels, action_dim=11):
        super(DuelingConvNet, self).__init__()
        self.name = 'DuelingConvNet'
        self.embedding = ConvNet(input_channels)

        self.advantage = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(128 * 5 * 5, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, 1)
        adv = self.advantage(x)
        state_value = self.value(x)
        return state_value + adv - adv.mean()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d or nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class DuelingResNet(nn.Module):
    def __init__(self, input_channels, action_dim=11, block=None, num_blocks=None):
        super(DuelingResNet, self).__init__()
        self.name = 'DuelingResNet'
        self.embedding = ResNet(input_channels, block, num_blocks)

        self.advantage = nn.Sequential(
            nn.Linear(256 * block.expansion * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.advantage = nn.Sequential(
            nn.Linear(256 * block.expansion * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, 1)
        adv = self.advantage(x)
        state_value = self.value(x)
        return state_value + adv - adv.mean()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d or nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    batch_vectors = torch.randn(8, 32, 6)
    batch_images = torch.randn((8, 32, 3, 255, 255))
    batch_next_vectors = torch.randn(8, 32, 6)
    batch_next_images = torch.randn((8, 32, 3, 255, 255))
    batch_action_idx = torch.randint(0, 11, (8, 32))
    batch_rewards = torch.rand(8, 32)

    fc_net = FCNet(vec_dim=6, output_dim=11, history_len=32)
    out = fc_net(vector=batch_vectors)
    # Image input
    # eval_net1 = Conv_LSTM(image_channels=3, action_dim=11, lstm_dim=512, lstm_layers=2)
    # target_net1 = Conv_LSTM(image_channels=3, action_dim=11, lstm_dim=512, lstm_layers=2)
    #
    # # Vector input
    # eval_net2 = FC_LSTM(vec_dim=6, action_dim=11, lstm_dim=512, lstm_layers=2)
    # target_net2 = FC_LSTM(vec_dim=6, action_dim=11, lstm_dim=512, lstm_layers=2)
    #
    # # Multi modal input
    # eval_net3 = Multimodal_LSTM(image_channels=3, vec_dim=6, action_dim=11, lstm_dim=512, lstm_layers=2)
    # target_net3 = Multimodal_LSTM(image_channels=3, vec_dim=6, action_dim=11, lstm_dim=512, lstm_layers=2)
    #
    # eval_lstm_tuple = (Variable(torch.zeros(2, 32, 512).float()),
    #                    Variable(torch.zeros(2, 32, 512).float()))
    # target_lstm_tuple = (Variable(torch.zeros(2, 32, 512).float()),
    #                      Variable(torch.zeros(2, 32, 512).float()))
    #
    # for i, (image, vector) in enumerate(zip(batch_images, batch_vectors)):
    #     q_evals, hidden = eval_net3(image, vector, eval_lstm_tuple)
    #     q_evals = q_evals.gather(1, batch_action_idx[i].reshape(-1, 1))
    #     q_next, target_lstm_tuple = target_net3(batch_next_images[i], batch_next_vectors[i], target_lstm_tuple)
    #     q_next = torch.max(q_next, 1)[0]
    #     q_target = batch_rewards[i] + 0.99 * q_next
    #     print(q_evals)
