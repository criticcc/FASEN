import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from src.utils import calculate_auc_aupr_f1
from torch import optim
from src.optim_utils import Lamb,Lookahead,LookaheadAdam
from src.optim import LRScheduler
import numpy as np

def train(model, optimizer, train_loader, config,c):
    """
    训练模型，使用训练数据的原始特征计算重建误差作为损失，并使用 LRScheduler 动态调整学习率。

    Args:
        model: WaveletAutoEncoder 模型
        optimizer: 优化器
        train_data: 训练数据 (多频率分解的张量)
        train_origine: 原始训练数据 (未分解频率的张量)
        c: 配置参数，用于初始化 LRScheduler。
    """

    device=config['device']
    # 初始化学习率调度器
    flag=0
    scheduler = LRScheduler(c=c, name=c.exp_scheduler, optimizer=optimizer)

    # 计算每个 epoch 的步数

    num_steps_per_epoch = 1
    max_epochs = int(np.ceil(c.exp_num_total_steps / num_steps_per_epoch))

    print(f"step_perepochis:{num_steps_per_epoch}")
    for epoch in range(max_epochs):
        model.train()  # 确保模型处于训练模式
        for step, (freq_batch, origine_batch, label_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # 前向传播
            if epoch==max_epochs-1:
                flag=1
                print('训练最后一轮，打印权重')

            freq_batch = [x.to(device) for x in freq_batch]  # list of [B, d]
            origine_batch = origine_batch.to(device)

            train_output = model(freq_batch,flag)

            # 计算重建误差 (MSE)
            mse = torch.norm(origine_batch - train_output, dim=1)
            loss = mse.mean()


            # 反向传播
            loss.backward()
            optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新学习率

        # 打印训练损失（每个 epoch 打印一次）
        print(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {loss.item():.6f}")

    # 保存模型
    torch.save(model, "model.pth")
    print(f"模型已保存至: model.pth")


def evaluate(test_loader, config,c):
    """
    测试模型性能，加载保存的模型进行评估。
    """
    # 加载保存的模型
    device = config['device']
    model = torch.load('model.pth')
    model.eval()

    all_mse = []
    all_labels = []

    with torch.no_grad():
        for freq_batch, origine_batch, label_batch in test_loader:
            freq_batch = [x.to(device) for x in freq_batch]
            origine_batch = origine_batch.to(device)
            label_batch = label_batch.to(device)

            test_output = model(freq_batch, 1)

            mse = torch.norm(origine_batch - test_output, dim=1)

            all_mse.append(mse.cpu())
            all_labels.append(label_batch.cpu())



    mse_all = torch.cat(all_mse).numpy()
    labels_all = torch.cat(all_labels).numpy()

    auc, aupr, f1 = calculate_auc_aupr_f1(labels_all, mse_all)
    print(f"AUC: {auc:.4f}, AUPR: {aupr:.4f}, f1: {f1:.4f}")
    return auc, aupr, f1


def init_optimizer(c, model_parameters, device):
    # 初始化基础优化器
    if 'default' in c.exp_optimizer:
        print("Using Adam optimizer")  # 调试信息
        optimizer = optim.Adam(params=model_parameters, lr=c.exp_lr)
    elif 'lamb' in c.exp_optimizer:
        print("Using Lamb optimizer")  # 调试信息
        lamb = Lamb
        optimizer = lamb(
            model_parameters, lr=c.exp_lr, betas=(0.9, 0.999),
            weight_decay=c.exp_weight_decay, eps=1e-6)
    else:
        raise NotImplementedError(f"Optimizer {c.exp_optimizer} not implemented")

    # 如果启用了 Lookahead 包装
    if c.exp_optimizer.startswith('lookahead_'):
        print("Using Lookahead optimizer")  # 调试信息
        optimizer = Lookahead(optimizer, k=c.exp_lookahead_update_cadence)

    # 输出调试信息
    print(f"Optimizer initialized: {type(optimizer)}")
    return optimizer

