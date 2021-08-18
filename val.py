import torch
from utils import accuracy, AverageMeter

def val(model, dataloader, epoch=9999):
    acc1_meter = AverageMeter(name='accuracy top 1')
    acc4_meter = AverageMeter(name='accuracy top 4')
    n_iters = len(dataloader)
    model.eval()
    with torch.no_grad():
        for iter_idx, (images, labels) in enumerate(dataloader):

            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            acc1, acc4 = accuracy(outputs, labels, topk=(1, 4))
            acc1_meter.update(acc1[0], images.shape[0])
            acc4_meter.update(acc4[0], images.shape[0])

            print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: \tAcc top-1 {acc1_meter.val:.2f}({acc1_meter.avg:.2f}) \tAcc top-4 {acc4_meter.val:.2f}({acc4_meter.avg:.2f})", end='\r')
    print("")
    print(f"Epoch {epoch} validation: top-1 acc {acc1_meter.avg} top-4 acc {acc4_meter.avg}")
    return acc1_meter.avg, acc4_meter.avg