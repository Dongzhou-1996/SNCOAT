import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Models.embedding_models import ConvNet
from Models.actor_models import DiscreteActorHead, ContinueActorHead
from Models.critic_models import CriticHead


class ActorCritic(nn.Module):
    def __init__(self, input_channels, output_dim, max_action=2, with_SE=True, vis=False, continuous=True):
        super(ActorCritic, self).__init__()
        self.embedding = ConvNet(input_channels, with_SE, vis)
        self.embedding_fc = nn.Linear(128 * 5 * 5, 1024)
        self.continuous = continuous
        if continuous:
            self.actor_head = ContinueActorHead(
                input_dim=1024, output_dim=output_dim, hidden_dim=128, max_action=max_action
            )
        else:
            self.actor_head = DiscreteActorHead(
                input_dim=1024, output_dim=output_dim, hidden_dim=128
            )

        self.critic_head = CriticHead(input_dim=1024, hidden_dim=128)

    def forward(self, image, vector, hidden=None):
        feature = torch.flatten(self.embedding(image), 1)
        feature = F.relu(self.embedding_fc(feature))
        value = self.critic_head(feature)
        if self.continuous:
            mu, sigma = self.actor_head(feature)
            return mu, sigma, value
        else:
            probs = self.actor_head(feature)
            return probs, value


    def actor(self, image, vector, hidden=None):
        feature = torch.flatten(self.embedding(image), 1)
        feature = F.relu(self.embedding_fc(feature))
        if self.continuous:
            mu, sigma = self.actor_head(feature)
            return mu, sigma
        else:
            probs = self.actor_head(feature)
            return probs

    def critic(self, image, vector, hidden=None):
        feature = torch.flatten(self.embedding(image), 1)
        feature = F.relu(self.embedding_fc(feature))
        value = self.critic_head(feature)
        return value


class ActorCriticLSTM(nn.Module):
    def __init__(self, input_channels, output_dim, lstm_dim, lstm_layers=2, with_SE=True, vis=False):
        super(ActorCriticLSTM, self).__init__()
        self.embedding = ConvNet(input_channels, with_SE, vis)
        self.embedding_fc = nn.Linear(128 * 5 * 5, 1024)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim, lstm_layers)
        self.actor_head = DiscreteActorHead(input_dim=1024, output_dim=output_dim, hidden_dim=128)
        self.critic_head = CriticHead(input_dim=1024, hidden_dim=128)

    def forward(self, image, vector, hidden=None):
        feature = torch.flatten(self.embedding(image), 1)
        feature = F.relu(self.embedding_fc(feature))
        recur_feature = F.relu(self.lstm(feature))
        value = self.critic_head(recur_feature)
        probs = self.actor_head(recur_feature)
        dist = Categorical(probs)
        return dist, value

    def actor(self, image, vector, hidden=None):
        feature = torch.flatten(self.embedding(image), 1)
        feature = F.relu(self.embedding_fc(feature))
        recur_feature = F.relu(self.lstm(feature))
        probs = self.actor_head(recur_feature)
        dist = Categorical(probs)
        return dist

    def critic(self, image, vector, hidden=None):
        feature = torch.flatten(self.embedding(image), 1)
        feature = F.relu(self.embedding_fc(feature))
        recur_feature = F.relu(self.lstm(feature))
        value = self.critic_head(recur_feature)
        return value

