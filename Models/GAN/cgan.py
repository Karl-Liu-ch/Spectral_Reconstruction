import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, num_input, num_output):
        super(Generator, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU())
            return layer

        self.Net = nn.Sequential(
            *Conv(num_input, 32),
            *Conv(32, 32),
            *Conv(32, num_output),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.Net(input)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, input_nums):
        super(Discriminator, self).__init__()
        def Conv(input_nums, output_nums):
            layer = []
            layer.append(nn.Conv2d(input_nums, output_nums, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
            layer.append(nn.BatchNorm2d(output_nums))
            layer.append(nn.ReLU(True))
            return layer

        self.Net = nn.Sequential(
            *Conv(input_nums, 64),
            *Conv(64, 128),
            *Conv(128, 256),
            *Conv(256, 512),
            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten()
        )

    def forward(self, input):
        output = self.Net(input)
        return output
    
class CGAN(nn.Module):
    def __init__(self) -> None:
        super().__init__()




    
if __name__ == '__main__':
    input = torch.randn([1, 31, 128, 128]).cuda()
    D = Discriminator(31).cuda()
    output = D(input)
    print(output.shape)