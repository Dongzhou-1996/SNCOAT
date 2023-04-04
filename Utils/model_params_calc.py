import os
import torch
from ptflops import get_model_complexity_info
# from Envs.SNCOAT_Env_v2 import DISCRETE_ACTIONS
from Models.critic_models import ConvQNet, ResQNet, BasicBlock, Bottleneck, LSTMQNet, LSTMAQNet

model_names = [
    'ConvNet',
    # 'ResNet18',
    # 'ResNet34',
    # 'ResNet50',
    'Conv-LSTM',
    'Conv-Attention-LSTM'
]
in_channels = 3
action_dim = 11

for i, model_name in enumerate(model_names):
    if model_name == 'ConvNet':
        eval_net = ConvQNet(input_channels=in_channels, action_dim=action_dim)
    elif model_name == 'ResNet18':
        eval_net = ResQNet(input_channels=in_channels, action_dim=action_dim,
                           block=BasicBlock, num_blocks=[2, 2, 2, 2])
    elif model_name == 'ResNet34':
        eval_net = ResQNet(input_channels=in_channels, action_dim=action_dim,
                           block=BasicBlock, num_blocks=[3, 4, 6, 3])
    elif model_name == 'ResNet50':
        eval_net = ResQNet(input_channels=in_channels, action_dim=action_dim,
                           block=Bottleneck, num_blocks=[3, 4, 6, 3])
    elif model_name == 'Conv-LSTM':
        eval_net = LSTMQNet(input_channels=in_channels, action_dim=action_dim, lstm_dim=256, lstm_layers=2)
    elif model_name == 'Conv-Attention-LSTM':
        eval_net = LSTMAQNet(input_channels=in_channels, action_dim=action_dim, lstm_dim=256, lstm_layers=2)
    else:
        raise ValueError('Unsupported model type')


    flops, params = get_model_complexity_info(eval_net, (3, 255, 255),
                                              as_strings=True, print_per_layer_stat=True)
    print('Model name: ' + model_name)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    print('=========================================================')

