import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


def test4opt():
    # 使用两个优化器，每个更新对应的变量
    x = torch.ones([2,3], requires_grad=True)
    y = torch.ones([2,3], requires_grad=True)

    # 设置2个优化器分别优化不同的参数
    opt1 = torch.optim.SGD([x], lr=0.01)
    opt2  = torch.optim.SGD([y], lr=0.01)

    # loss分别需要detach其他的变量
    l1 = (x*y.detach()).sum()
    l2 =( x.detach()*y).sum()

    # 清空优化器相关的变量
    opt1.zero_grad()
    opt2.zero_grad()
    # 【同时】反向传播
    l1.backward()
    l2.backward()
    # [同时]更新
    opt1.step()
    opt2.step()


def test4opt2():
    # 仅使用一个优化器也能解决，仅需要将x,ydetach即可
    x = torch.ones([2,3], requires_grad=True)
    y = torch.ones([2,3], requires_grad=True)

    # 设置2个优化器分别优化不同的参数
    opt1 = torch.optim.SGD([x,y], lr=0.01)
    for i in range(5):
        # loss分别需要detach其他的变量
        l1 = (x*y.detach()).sum()
        l2 =( x.detach()*y).sum()

        # 清空优化器相关的变量
        opt1.zero_grad()
        # 【同时】反向传播
        l1.backward()
        l2.backward()
        # [同时]更新
        opt1.step()


def test4task_dec():
    '''测试任务分配代码是否合理
    '''
    x = torch.ones([2,3], requires_grad=True)
    y = torch.ones([2,3], requires_grad=True)

    y[1,:] =0  #: 不能够任意赋值
    q_list = []
    for i in range(2):
        q=torch.zeros(x.shape)

        mask = y==i
        q[mask] = x[mask]
        q_list.append(q)
    loss = 0
    for i in range(2):
        loss += q_list[i].sum()

    opt1 = torch.optim.SGD([x], lr=0.01)
    opt1.zero_grad()
    loss.backward()
    opt1.step()


if __name__=='__main__':
    test4opt2()