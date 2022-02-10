import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from skimage.segmentation import slic
from torchvision import transforms

from instance_conv import InstanceConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VanillaNet(nn.Module):
    def __init__(self):
        super(VanillaNet, self).__init__()
        self.conv1 = InstanceConv(3, 8, kernel_size=3, stride=1, padding=1)
        self.seq1 = nn.Sequential(
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.conv2 = InstanceConv(8, 16, kernel_size=3, stride=1, padding=1)
        self.seq2 = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.depth_pred = InstanceConv(16, 1, kernel_size=1)

    def forward(self, x, mask):
        out, out_mask = self.conv1(x, mask)
        out = self.seq1(out)

        out, out_mask = self.conv2(out, out_mask)
        out = self.seq2(out)

        pred, _ = self.depth_pred(out, out_mask)

        return pred


if __name__ == "__main__":
    image = np.array(Image.open("input.png"))
    mask = slic(image, n_segments=64, sigma=1, start_label=1)
    image = data_transform(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0).to(device)
    target = (
        torch.from_numpy(np.load("gt.npy")).float().unsqueeze(0).unsqueeze(0).to(device)
    )

    net = VanillaNet().to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(net.parameters())

    iteration = 1000
    running_loss = 0.0
    for i in range(iteration):
        optimizer.zero_grad()

        pred = net(image, mask)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 50 == 49: 
            print(f"[{i + 1:5d}] loss: {running_loss / 50:.3f}")
            running_loss = 0.0
