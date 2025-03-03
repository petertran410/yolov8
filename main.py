from ultralytics import YOLO
import torch.nn as nn

# Define a simplified C2F class


class C2F(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride_num, shortcut=True):
        super(C2F, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride_num, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.blocks = nn.Sequential(
            *[Bottleneck(out_channels, out_channels, shortcut=shortcut) for _ in range(num_blocks)]
        )

        self.conv2 = nn.Conv2d(out_channels * (1 + num_blocks), out_channels,
                               kernel_size=3, stride=stride_num, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        y1 = self.act1(self.bn1(self.conv1(x)))
        y2 = self.blocks(y1)
        y = torch.cat((y1, y2), dim=1)
        return self.act2(self.bn2(self.conv2(y)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU(inplace=True)

        self.shortcut = shortcut

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.act2(self.bn2(self.conv2(y)))
        if self.shortcut:
            y = y + x
        return y


model = YOLO('/kaggle/working/yolov8m.pt')

model.model.model[0] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2,
    stride_num=2
)
model.model.model[1] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2,
    stride_num=2
)
model.model.model[2] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2,
    stride_num=2
)
model.model.model[3] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2,
    stride_num=1
)
model.model.model[4] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[5] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[6] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[7] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[8] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[9] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[10] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[11] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[12] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[13] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[14] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[15] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[16] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[17] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=4, stride_num=1
)
model.model.model[18] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[19] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[20] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[21] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)
model.model.model[22] = C2F(
    in_channels=4,
    out_channels=32,
    num_blocks=2, stride_num=1
)

print(nn.Sequential(*model.model.model))

model.model.model = nn.Sequential(*model.model.model)

path = "/kaggle/input/warp-d/Warp-D/data.yaml"

results = model.train(
    data=path,
    epochs=300,
    imgsz=640,
    batch=16,
    lr0=0.0005,
    lrf=0.0005,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    patience=0,
    conf=0.2,
    iou=0.5,
    box=8,
    cls=0.8,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    auto_augment='randaugment',
    optimizer='AdamW',
    save=True,
    verbose=True,
    exist_ok=True,
    name="Test on Warp-D dataset version 2",
    device='0,1'
)
