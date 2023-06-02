import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    # input shape: [batch_size, 1, 28, 28]
    # output shape: [batch_szie, 10]
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.train(False)

    def forward(self, x):
        # input shape is 1x1x28x28
        # Max pooling over a (2, 2) window, if use default stride, error will happen
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), stride=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), stride=(2, 2))
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNN(nn.Module):
    # input shape: [sequence_len, batch_size, input_size]
    # output shape: [batch_size, output_size], [batch_size, hidden_size]
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        torch.manual_seed(2)
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.hidden = torch.zeros(1, self.hidden_size)
        self.train(False)

    def forward(self, input):
        hidden = self.hidden
        for i in range(input.size(0)):
            combined = torch.cat((input[i], hidden), 1)
            hidden = torch.sigmoid(self.i2h(combined))
            output = self.i2o(combined)
        # return output, hidden
        return output


class LSTMCell(nn.Module):
    # input shape: [batch_size, input_size], [batch_size, hidden_size]
    # output shape: [batch_size, hidden_size], [batch_size, hidden_size]
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.i2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2i = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2c = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden, cell):
        combined = torch.cat((input, hidden), 1)
        forget_gate = torch.sigmoid(self.i2f(combined))
        input_gate = torch.sigmoid(self.i2i(combined))
        output_gate = torch.sigmoid(self.i2o(combined))
        cell_gate = torch.tanh(self.i2c(combined))
        cell = forget_gate * cell + input_gate * cell_gate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def initCell(self):
        return torch.zeros(1, self.hidden_size)


class LSTM(nn.Module):
    # input shape: [sequence_size, batch_size, input_size]
    # output shape: [batch_size, output_size], [batch_size, hidden_size]
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = self.lstm.initHidden()
        cell = self.lstm.initCell()
        for i in range(input.size(0)):
            hidden, cell = self.lstm(input[i], hidden, cell)
        output = self.fc(hidden)
        return output


class GRUCell(nn.Module):
    # input shape: [batch_size, input_size], [batch_size, hidden_size]
    # output shape: [batch_size, hidden_size]
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.i2r = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2z = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2n = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        reset_gate = torch.sigmoid(self.i2r(combined))
        update_gate = torch.sigmoid(self.i2z(combined))
        new_gate = torch.tanh(self.i2n(torch.cat((input, reset_gate * hidden), 1)))
        hidden = (1 - update_gate) * hidden + update_gate * new_gate
        return hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class GRU(nn.Module):
    # input shape: [sequence_size, batch_size, input_size]
    # output shape: [batch_size, output_size], [batch_size, hidden_size]
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        hidden = self.gru.initHidden()
        for i in range(input.size(0)):
            hidden = self.gru(input[i], hidden)
        output = self.fc(hidden)
        return output

class Inception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.conv5 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv5(F.relu(self.conv2(x))))
        x3 = F.relu(self.conv6(F.relu(self.conv3(x))))
        x4 = F.relu(self.conv4(self.pool(x)))
        return torch.cat([x1, x2, x3, x4], dim=1)

 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.inception1 = Inception(192, 256)
        self.inception2 = Inception(256, 480)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3 = Inception(480, 512)
        self.inception4 = Inception(512, 512)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.pool(self.inception3(x))
        x = self.inception4(x)
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Test the models

    # input_size = 10
    # hidden_size = 20
    # sequence_len = 3
    # output_size = 5
    # batch_size = 1

    # input = torch.randn(batch_size, 1, 28, 28)
    # lenet = LeNet()
    # output = lenet(input)
    # print("LeNet output: ", output)

    # input = torch.randn(sequence_len, batch_size, input_size)
    # hidden = torch.randn(1, hidden_size)

    # rnn = RNN(input_size, hidden_size, output_size)
    # output, hidden = rnn(input)
    # print("RNN output: ", output)

    # lstm = LSTM(input_size, hidden_size, output_size)
    # output, hidden = lstm(input)
    # print("LSTM output: ", output)

    # gru = GRU(input_size, hidden_size, output_size)
    # output, hidden = gru(input)
    # print("GRU output: ", output)

    batch_size = 1
    channels = 3
    height = 224
    width = 224
    
    inception = Net()
    input = torch.randn(batch_size, channels, height, width)
    output = inception(input)
    print(output)
