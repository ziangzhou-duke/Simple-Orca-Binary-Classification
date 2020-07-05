import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAttenGRU(nn.Module):
    def __init__(self, classes):
        super(SimpleAttenGRU, self).__init__()
        self.feature_dim = 40
        self.hidden_dim = 128
        self.output_dim = classes
        self.num_layers = 2
        self.gru = nn.GRU(self.feature_dim, self.hidden_dim, self.num_layers, dropout=0.2)
        self.fc = nn.Linear(self.hidden_dim, classes)
        self.softmax = nn.Softmax(dim=1) 
        self.tanh = nn.Tanh()
        # atten parameter
        self.weight_proj = nn.Parameter(torch.Tensor(self.hidden_dim, 1))
        self.weight_W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_dim,1))
        self.weight_proj.data.uniform_(-0.1, 0.1)
        self.weight_W.data.uniform_(-0.1, 0.1)
        self.bias.data.zero_()

    def batch_soft_atten(self, seq, W, bias, v):
        s = []
        batch_size = seq.shape[1]
        bias_dim = bias.size()
        for i in range(seq.size(0)):
            _s = torch.mm(seq[i], W)
            _s_bias = _s + bias.expand(bias_dim[0], batch_size).transpose(0,1)
            _s_bias = torch.tanh(_s_bias)
            _s = torch.mm(_s_bias, v)
            s.append(_s)
        s = torch.cat(s, dim=1)
        soft = self.softmax(s)
        return soft
    
    def attention_mul(self, rnn_outputs, att_weights):
        attn_vectors = []
        for i in range(rnn_outputs.size(0)):
            h_i = rnn_outputs[i]
            a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
            h_i = a_i * h_i
            h_i = h_i.unsqueeze(0)
            attn_vectors.append(h_i)
        attn_vectors = torch.cat(attn_vectors, dim=0)
        return torch.sum(attn_vectors, 0)

    def soft_attention(self, ht):
        atten_alpha = self.batch_soft_atten(ht, self.weight_W, self.bias, self.weight_proj)
        atten_vects = self.attention_mul(ht, atten_alpha.transpose(0, 1))
        return atten_vects

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[2], x.shape[3])
        x = x.transpose(0, 1) # batch second
        self.gru.flatten_parameters()
        # self.hidden = self.init_hidden(x.shape[1])
        x, ht = self.gru(x)
        x = self.soft_attention(x)
        x = self.fc(x)
        return x, x

def test():
    data = torch.rand(256, 1, 100, 40)
    net = SimpleAttenGRU(4)
    result, _ = net(data)
    print(result.shape)

if __name__ == "__main__":
    test()
