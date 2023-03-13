import math
import torch
import torch.nn as nn


class MemoryEncoder(torch.nn.Module):
    """
    As originally defined in the paper, the encoder architecture is
    Conv2(1, 2, 16) -> Conv2(3, 2, 32) -> Conv2(3, 2, 64)
    where Conv2(k, s, c) means a 2D convolution with kernel size k, stride s and c output channels.
    """
    def __init__(self):
        super(MemoryEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # flatten the output
        z = out.reshape(out.size(0), -1)
        return z


class MemoryDecoder(nn.Module):
    """
    As originally defined in the paper, the decoder architecture is
    Dconv2(3, 2, 64) -> Dconv2(3, 2, 32) -> Dconv2(3, 2, 1)
    where Dconv2(k, s, c) means a 2D transposed convolutional layer
    with kernel size k, stride s and c output channels.
    """
    def __init__(self):
        super(MemoryDecoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2,  output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.layer3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, output_padding=1, padding=1)

    def forward(self, x):
        # reshape the input
        out = x.reshape(x.size(0), 64, 2, 2)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class MemoryAutoEncoder(nn.Module):
    def __init__(self, n=100, c=400):
        super(MemoryAutoEncoder, self).__init__()
        self.encoder = MemoryEncoder()
        self.decoder = MemoryDecoder()
        self.N = n
        self.C = c
        # Contruct and initialize the memory
        init_memory = torch.zeros(self.N, self.C)
        nn.init.kaiming_normal_(init_memory)
        self.memory = nn.Parameter(init_memory, requires_grad=True)

        # Distance function
        self.d = nn.CosineSimilarity(dim=2, eps=1e-6)
        # From the paper: "In practice, setting the theshold /lambda as a value
        # in the interval [1/N, 3/N], can reder desirable results."
        self.lambda_thr = 1.0 / self.N
        self.eps = 1e-12
        self.shrinkage = True

    def forward(self, x):
        z = self.encoder(x)
        # Compute the weights w_i
        # 1. Calculate d(z, m_i) for each memory slot m_i, Eq. 5 in paper
        d = self.d(z.unsqueeze(1), self.memory.unsqueeze(0))
        # 2. Calculate softmax over the distances, Eq. 4 in paper
        w = nn.functional.softmax(d, dim=1)
        if self.shrinkage:
            # 3. Hard Shrinkage for Sparse Addressing, Eq. 7 in paper
            w_hat = nn.functional.relu(w - self.lambda_thr) / (torch.abs(w - self.lambda_thr) + self.eps)
            #print(f"w_hat: {w.max()}, lambda: {self.lambda_thr}")
            w_hat = w_hat * w
            # Normalize w_hat with L1 norm wi = wi / ||w||_1
            w_hat = w_hat / (torch.sum(torch.abs(w_hat), dim=1, keepdim=True) + self.eps)
        else:
            w_hat = w
        # 4. Calculate zhat
        z_hat = w_hat@self.memory
        # 5. Decode
        x_hat = self.decoder(z_hat)

        return x_hat, w_hat
