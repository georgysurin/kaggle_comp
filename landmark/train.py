import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import time
from datetime import timedelta

from datatset import LandmarkDataset
from model import MyResNet
from arc_loss import ArcMarginProduct
from focal_loss import FocalLoss
from utils import print_one_line, set_lr

parallel_mode = 'ModelParallel'
gpu_num = torch.cuda.device_count()
print('gpu_num', gpu_num)

df = pd.read_csv('/home/iris/formemorte/landmark/train.csv')
dataset = LandmarkDataset(df)

model = MyResNet()
model = nn.DataParallel(model)
model.cuda()

num_chunk_margin_fc = gpu_num
arc_loss_chunks = nn.ModuleList()
start_index = 0
sum = 0
for i in range(num_chunk_margin_fc):
    end_label = int(((81313 - 1) / num_chunk_margin_fc) * (i + 1))

    start_label = start_index

    print(end_label)
    class_num = end_label - start_label + 1

    chunk = ArcMarginProduct(512, class_num, s=8, m=0.5,
                             start_label=start_label, end_label=end_label,
                             easy_margin=True, parallel_mode=parallel_mode)

    sum += chunk.weight.size(0)
    arc_loss_chunks.append(chunk.cuda(i))
    start_index = end_label + 1

print(sum)
criterion = FocalLoss(gamma=2)
criterion.cuda()

optimizer_param = [{'params': _chunk.parameters()} for _chunk in arc_loss_chunks]
optimizer_param += [{'params': model.parameters()}]

lr = 1e-3
momentum = 0.9
# optimizer = torch.optim.SGD(optimizer_param, lr=lr, momentum=momentum, weight_decay=5e-4)
optimizer = torch.optim.Adam(optimizer_param, lr=lr)

dataloader = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=4, pin_memory=True)

for epoch in range(1, 1001):
    if epoch in [10, 20, 30]:
        lr *= 0.1
        set_lr(optimizer, lr)

    start_time = time.time()

    train_accuracy = 0
    train_loss = 0
    total = 0
    correct = 0
    batch_idx = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        x = model(inputs)

        x_list = []
        for i in range(num_chunk_margin_fc):
            _x = arc_loss_chunks[i](x.cuda(i), targets.cuda(i))
            x_list.append(_x.cuda(0))
        outputs = torch.cat(x_list, dim=1)

        loss = criterion(outputs, targets)
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        batch_idx += 1

        train_accuracy = float(100.0 * correct) / total
        print_one_line('Epoch {} Loss {:.4f} | Acc={:.2f}% ({}/{}/{})'.format(epoch,
                                                                           train_loss / batch_idx,
                                                                           train_accuracy,
                                                                           correct, total, len(dataset)))
    print('')
    elapsed_time_secs = time.time() - start_time

    msg = "Epoch took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))

    print(msg)

    torch.save(model.module.state_dict(), 'resnet50_arc_gem.pth')
    for i, _chunk in enumerate(arc_loss_chunks):
        torch.save(_chunk.state_dict(), f'arcloss_chukn_{i}.pth')
