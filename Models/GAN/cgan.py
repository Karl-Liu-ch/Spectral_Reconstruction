import torch.nn as nn

class Generator_28(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator_28, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.ConvTranspose2d(input_nums, output_nums, kernel_size=(4,4), stride=(2,2), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(num_input, 1024),
            *Conv(1024, 512),
            *Conv(512, 256),
            nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, num_output, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.Net(input)
        return output