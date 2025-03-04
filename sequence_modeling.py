import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output




class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向GRU输出

    def forward(self, x):
        # 初始化隐藏状态
        if len(x.size()) == 2:  # 如果输入是2D张量，则扩展为3D
            x = x.unsqueeze(0)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirection
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out = self.fc(out)  # 最后一时间步的输出
        return out


'''
def test():
    Model = nn.Sequential(BidirectionalLSTM(input_size=192,hidden_size=256,output_size=256),
                          BidirectionalLSTM(input_size=256,hidden_size=256,output_size=256),)
    x = torch.randn(size=(48, 28, 192))
    preds = Model(x)
    y = torch.randn(size=(48, 28, 256))
    z = torch.cat([x,y],2)
    print(f"preds shape if {z.shape}")


if __name__ == "__main__":
    test()
'''