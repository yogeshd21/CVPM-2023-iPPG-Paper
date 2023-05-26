import torch
import torch.nn as nn

tr = torch

# -------------------------------------------------------------------------------------------------------------------
# DeepPhys network
# -------------------------------------------------------------------------------------------------------------------
class DeepPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask1 = None
        self.mask2 = None

        # Appearance stream
        self.a_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn1 = nn.BatchNorm2d(32)

        self.a_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.a_bn2 = nn.BatchNorm2d(32)
        self.a_d1 = nn.Dropout2d(p=0.50)

        self.a_softconv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.a_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.a_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_bn3 = nn.BatchNorm2d(64)

        self.a_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.a_bn4 = nn.BatchNorm2d(64)
        self.a_d2 = nn.Dropout2d(p=0.50)
        self.a_softconv2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Motion stream
        self.m_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_bn1 = nn.BatchNorm2d(32)
        self.m_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.m_bn2 = nn.BatchNorm2d(32)
        self.d1 = nn.Dropout2d(p=0.50)

        self.m_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.m_bn3 = nn.BatchNorm2d(64)
        self.m_conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.m_bn4 = nn.BatchNorm2d(64)
        self.d2 = nn.Dropout2d(p=0.50)
        self.m_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Fully conected blocks
        self.d3 = nn.Dropout(p=0.25)
        self.fully1 = nn.Linear(in_features=64 * 9 * 9, out_features=128, bias=True)
        #self.fully1 = nn.Linear(in_features=64 * 256 * 256, out_features=128, bias=True)
        self.fully2 = nn.Linear(in_features=128, out_features=1, bias=True)

    def forward(self, A, M):
        # (A) - Appearance stream -------------------------------------------------------------
        # First two convolution layer
        A = tr.tanh(self.a_bn1(self.a_conv1(A)))
        A = tr.tanh(self.a_bn2(self.a_conv2(A)))
        A = self.a_d1(A)

        # Calculating attention mask1 with softconv1
        mask1 = tr.sigmoid(self.a_softconv1(A))
        B, _, H, W = A.shape
        norm = 2 * tr.norm(mask1, p=1, dim=[1, 2, 3])
        norm = norm.reshape(B, 1, 1, 1)
        mask1 = tr.div(mask1 * H * W, norm)
        self.mask1 = mask1

        # Pooling
        A = self.a_avgpool(A)
        # Last two convolution
        A = tr.tanh(self.a_bn3(self.a_conv3(A)))
        A = tr.tanh(self.a_bn4(self.a_conv4(A)))
        A = self.a_d2(A)

        # Calculating attention mask2 with softconv2
        mask2 = tr.sigmoid(self.a_softconv2(A))
        B, _, H, W = A.shape
        norm = 2 * tr.norm(mask2, p=1, dim=[1, 2, 3])
        norm = norm.reshape(B, 1, 1, 1)
        mask2 = tr.div(mask2 * H * W, norm)
        self.mask2 = mask2

        # (M) - Motion stream --------------------------------------------------------------------
        M = tr.tanh(self.m_bn1(self.m_conv1(M)))
        M = self.m_bn2(self.m_conv2(M))
        M = tr.tanh(tr.mul(M, mask1))  # multiplying with attention mask1
        M = self.d1(M)  # Dropout layer 1
        # Pooling
        M = self.m_avgpool1(M)
        # Last convs
        M = tr.tanh(self.m_bn3(self.m_conv3(M)))
        M = self.m_bn4(self.m_conv4(M))
        M = tr.tanh(tr.mul(M, mask2))  # multiplying with attention mask2
        M = self.d2(M)  # Dropout layer 2
        M = self.m_avgpool2(M)

        # (F) - Fully connected part -------------------------------------------------------------
        # Flatten layer out
        out = tr.flatten(M, start_dim=1)  # start_dim=1 to handle batches
        out = self.d3(out)  # dropout layer 3
        out = tr.tanh(self.fully1(out))
        out = self.fully2(out)

        return out

if __name__ == '__main__':
    def deepphys_test():
        model = DeepPhys()
        A = tr.randn(1, 3, 36, 36)
        out = model(A, A)
        print(out.shape, out)

    deepphys_test()               # OK!