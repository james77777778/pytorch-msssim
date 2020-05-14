import os
import sys
from pytorch_msssim import SSIM
import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

s = SSIM(data_range=1.)

a = (torch.randint(0, 255, size=(20, 3, 256, 256), dtype=torch.float32)
     .cuda()
     / 255.)
b = a * 0.5
a.requires_grad = True
b.requires_grad = True

start_record = torch.cuda.Event(enable_timing=True)
end_record = torch.cuda.Event(enable_timing=True)

start_record.record()
for _ in range(500):
    loss = s(a, b)
    loss.backward()
end_record.record()
torch.cuda.synchronize()

print('cuda time: ', start_record.elapsed_time(end_record)/1000)
