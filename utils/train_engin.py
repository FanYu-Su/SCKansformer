import sys

import torch
from tqdm import tqdm

from utils.distrubute_utils import is_main_process, reduce_value
from utils.lr_methods import warmup


def train_one_epoch(model, optimizer, data_loader, device, epoch, use_amp=False, lr_method=None):
    """
    训练一个 epoch 的函数。

    Args:
        model (nn.Module): 要训练的模型。
        optimizer (torch.optim.Optimizer): 优化器。
        data_loader (DataLoader): 数据加载器，用于迭代训练数据。
        device (torch.device): 训练设备。
        epoch (int): 当前训练的 epoch 数。
        use_amp (bool, optional): 是否使用自动混合精度训练，默认为 False。
        lr_method (callable, optional): 学习率调度方法，默认为 None。

    Returns:
        tuple: 包含训练损失和准确率的元组。
    """
    model.train()  # 开启训练模式
    loss_function = torch.nn.CrossEntropyLoss()  # 定义损失函数
    # 新建一个形状为1的张量，用于存储loss，用张量表示方便后续在GPU上进行同步处理；后续的reduce_value需要使用tensor；这里只说了是一维，没有说道是几行几列，在后续处理中，可以看出，他就是1*1的一个张量，说白了就是一个数值；
    train_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)
    # 以上是在存储acc和loss
    optimizer.zero_grad()

    lr_scheduler = None  # 占用位置，启动开关
    if epoch == 0 and lr_method == warmup:  # 判定是不是刚开始学习，如果是即epoch为0，然后lr_method是warmup的话，就使用warmup学习率调整方法，如下就可以进行warmup策略
        warmup_factor = 1.0/1000
        # 初始化学习率比例 lr=base_lr * warmup_factor，这里是1/1000，表示初始学习率是基础学习率的1/1000
        warmup_iters = min(1000, len(data_loader) - 1)
        # warmup_iters表示在多少个iter内进行warmup，这里取1000和len(data_loader)-1的最小值，防止数据集过小导致的iter数量不够1000的情况

        lr_scheduler = warmup(optimizer, warmup_iters, warmup_factor)
        # 本质是创建一个，在前warmup_iters个step中，lr从base_lr*warmup_factor线性增加到base_lr的一个调度器

    # 如果是主进程，则使用tqdm显示进度条，因为我们使用了分布式训练（即多张显卡同时训练），所以只有主进程负责打印日志信息，其中每张显卡都是一个进程
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 创建一个梯度缩放标量(GradScaler)，以最大程度避免使用fp16进行运算时的梯度下溢
    enable_amp = use_amp and "cuda" in device.type
    scaler = torch.cuda.amp.GradScaler(enabled=enable_amp)
    # 开启自动混合精度训练

    sample_num = 0  # 后续用于计算准确率，这里是保存总共有多少个样本
    # 从data_loader中获取data数据和处于第几个epoch中的step
    # 特别的：enumerate(data_loader)会返回两个值，第一个是索引（step），第二个是data_loader中的数据（data）；这里的data是一个batch的数据，包含了images和labels的batch数据；
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]  # 获取images的第0维度大小即batch_size，逐步累加得到总的样本数量
        # 以下是在使用autocast来进行前向传播和计算损失；pred是模型的输出，进一步的用pred和labels计算loss；
        with torch.amp.autocast('cuda', enabled=enable_amp):
            pred = model(images.to(device))  # 将images数据放到对应的device上再使用model进行计算
            # 已知得到的pred是images经过model在device上计算得出的，所以计算loss的时候，必须要把label也送到device上才能计算loss；
            loss = loss_function(pred, labels.to(device))

            pred_class = torch.max(pred, dim=1)[1]  # 取pred中每一行的最大值对应的索引作为预测类别；
            acc_num += torch.eq(pred_class, labels.to(device)).sum()  # 计算预测正确的样本数量，并累加到acc_num中；注意，这些数据都在device上进行计算；

        scaler.scale(loss).backward()  # 梯度缩放防止下溢
        scaler.step(optimizer)  # 更新参数
        scaler.update()  # 更新缩放因子，因为每次算出来的梯度可能不一样，需要动态调整缩放因子，才能恰好防止下溢
        # 以上是amp的反向传播和优化器更新的特有写法，正常写是：loss.backward(); optimizer.step();
        optimizer.zero_grad()  # 由于计算梯度在pytorch中是累加制，所以我们这里用的到的优化器每次更新完参数后都需要将梯度清零，避免被累加；

        # 这里的reduce_value是将多个GPU的loss同步，取各GPU的平均，单卡的时候相当于没有变化；detach是为了将计算图分离，防止内存泄漏；注意，这里做了累加
        train_loss += reduce_value(loss, average=True).detach()

        # 在进程中打印平均loss
        if is_main_process():
            info = '[epoch{}]: learning_rate:{:.5f}'.format(
                epoch + 1,
                optimizer.param_groups[0]["lr"]
            )
            data_loader.desc = info  # tqdm 成员 desc

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        if lr_scheduler is not None:  # 如果使用warmup训练，逐渐调整学习率
            lr_scheduler.step()

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    return train_loss.item() / (step + 1), acc_num.item() / sample_num
    # 这里的train_loss是所有batch的loss的一个张量，通过item得到他们的累加，所以要除以(step+1)得到平均loss；acc_num是预测正确的样本数量，除以总的样本数量sample_num得到准确率；


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # 验证集样本个数
    num_samples = len(data_loader.dataset)
    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred_class = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred_class, labels.to(device)).sum()

    # 等待所有进程计算完毕
    if device != torch.device('cpu'):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)
    val_acc = sum_num.item() / num_samples

    return val_acc
