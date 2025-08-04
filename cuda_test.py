import torch
import torchvision
boxes = torch.tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]], device='cuda')
scores = torch.tensor([0.5, 0.7], device='cuda')
print(torchvision.ops.nms(boxes, scores, 0.5))

