import torch


def accuracy(outputs, targets, topk=(1,)):
    """
    计算 Top-k 准确率
    
    参数:
        outputs: [batch_size, num_classes] 模型输出
        targets: [batch_size] 真实标签
        topk: tuple, 要计算的 k 值
    
    返回:
        list: 每个 k 对应的准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        
        return res