import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AddictiveAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, memory_dim=128):
        super(AddictiveAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, padding=0, bias=False)
        self.linear1 = nn.Linear(memory_dim, hidden_dim)

        self.attention_conv = nn.Conv2d(hidden_dim, 1, 1, 1, padding=0, bias=False)

    def feature_align(self, x: torch.Tensor, h: torch.Tensor):
        """
        Args:
            x: BxCxHxW
            h: BxN

        Returns: x2, x3, h3

        """
        x3 = self.conv1(x)  # BxDxHxW
        h3 = self.linear1(h).expand(*x3.shape[-2:], -1, -1).permute(2, 3, 0, 1)  # BxDxHxW
        return x3, h3

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        Args:
            x: BxCxHxW
            h: BxN

        Returns: a3, context

        """
        x2 = torch.flatten(x, start_dim=2)  # BxCx(H*W)

        x3, h3 = self.feature_align(x, h)  # BxDxHxW, BxDxHxW
        a1 = torch.tanh(x3 + h3)  # BxDxHxW
        a2 = torch.flatten(self.attention_conv(a1), start_dim=1)  # Bx(H*W)
        a3 = F.softmax(a2, dim=1)  # Bx(H*W)
        a4 = a3.expand(x2.shape[1], -1, -1).permute(1, 0, 2)  # BxCx(H*W)

        context = torch.sum(x2 * a4, dim=-1)  # BxC

        return a3, context


class DotProdAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, memory_dim=128):
        super(DotProdAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, padding=0)
        self.linear1 = nn.Linear(memory_dim, hidden_dim)

        self.attention_conv = nn.Conv2d(hidden_dim, 1, 1, 1, padding=0)

    def feature_align(self, x: torch.Tensor, h: torch.Tensor):
        """
        Args:
            x: BxCxHxW
            h: BxN

        Returns: x2, x3, h3

        """
        x3 = self.conv1(x)  # BxDxHxW
        h3 = self.linear1(h).expand(*x3.shape[-2:], -1, -1).permute(2, 3, 0, 1)  # BxDxHxW
        return x3, h3

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        Args:
            x: BxCxHxW
            h: BxN

        Returns: a3, context

        """
        x1, h1 = self.feature_align(x, h)  # BxDxHxW, BxDxHxW
        similarity = F.softmax(torch.sum(x1 * h1 / math.sqrt(256), dim=1), dim=1).unsqueeze(1)  # Bx1xHxW

        context = torch.sum(x * similarity, dim=[2, 3])  # BxC

        return similarity, context


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, x, mask=None):
        x = x.view(*x.shape[:2], -1).transpose(-1, -2)
        q, k, v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


if __name__ == '__main__':
    attention = MultiHeadAttention(128, head_num=4)
    x = torch.randn(32, 128, 5, 5)
    h = torch.randn(32, 512)
    context = attention(x)
    print(context)
