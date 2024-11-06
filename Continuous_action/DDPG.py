"""
Created on  Feb 28 2021
@author: wangmeng

深度确定性策略梯度
​ 在连续控制领域，比较经典的强化学习算法就是深度确定性策略梯度（deep deterministic policy gradient，DDPG）。DDPG的特点可以从名字当中拆解后取理解。拆解成深度、确定性和策略梯度。 深度是用了神经网络；确定性表示DDPG输出的是一个确定性的动作，可以用于连续动作的场景；策略梯度代表用到策略网络。


入门篇—DDPG代码逐行分析（pytorch）
在上一篇中我们简单整理了一下DQN的代码，这一篇则是解决连续状态，连续动作的问题----DDPG算法

一些需要注意的点
这里使用了OU-noise，由于其参数较多，调试起来较为复杂，在仿真中也可以使用简单的高斯噪声代替。至于为什么原论文要使用Ornstein-Uhlenbeck噪声，小伙伴们可以看知乎上强化学习中Ornstein-Uhlenbeck噪声是鸡肋吗？一文。
简单来说，相比于独立噪声，OU噪声适合于惯性系统，尤其是时间离散化粒度较小的情况，此外，它可以保护实际系统，如机械臂

优化了代码框架，修正了一些小错误。不过DDPG毕竟是16年提出的算法，只能说是拿来入门使用，在实际项目中我们还需要一些更优秀的算法。因此之后打算更新一些做项目使用的DRL算法，最后会将所有代码上传到我的gihub中

原文链接：https://blog.csdn.net/qq_37395293/article/details/114226081


案例: 倒立摆问题。钟摆以随机位置开始，目标是将其向上摆动，使其保持直立。 测试环境： Pendulum-v1
动作：往左转还是往右转，用力矩来衡量，即力乘以力臂。范围[-2,2]：（连续空间）
状态：cos(theta), sin(theta) , thetadot。
奖励：越直立拿到的奖励越高，越偏离，奖励越低。奖励的最大值为0。


"""
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from gym import Env
from torch.distributions import Normal
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs:int, num_actions:int, hidden_size:int, init_w = 3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w) # 均匀分布
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x) # 注意最后一层没有激活值
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs:int, num_actions:int, hidden_size:int, init_w = 3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # uniform_将tensor用从均匀分布中抽样得到的值填充。参数初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        #也用用normal_(0, 0.1) 来初始化的，高斯分布中抽样填充，这两种都是比较有效的初始化方式
        self.linear3.bias.data.uniform_(-init_w, init_w)
        #其意义在于我们尽可能保持 每个神经元的输入和输出的方差一致。
        #使用 RELU（without BN） 激活函数时，最好选用 He 初始化方法，将参数初始化为服从高斯分布或者均匀分布的较小随机数
        #使用 BN 时，减少了网络对参数初始值尺度的依赖，此时使用较小的标准差(eg：0.01)进行初始化即可

        #但是注意DRL中不建议使用BN

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0,0]

class OUNoise(object):
    """
    Ornstein-Uhlenbeck噪声
    """

    def __init__(self, action_space, mu=0.0, theta = 0.15, max_sigma = 0.3, min_sigma = 0.3, decay_period = 100000):#decay_period要根据迭代次数合理设置
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) *self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta* (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    """
    维纳过程的公式
    """
    def get_action(self, action:float, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayBuffer:
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state:np.array, action:np.array, reward:float, next_state:np.array, done:bool):
        # state:[state_dim, ]
        # action:[action_dim, ]
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity # 循环覆盖

    def clear(self):
        self.buffer.clear()
        self.position=0

    def sample(self, batch_size:int):
        batch = random.sample(self.buffer, batch_size) # random.sample 从列表中随机获取指定长度batch个样本
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# 继承gym.ActionWrapper，对action进行归一化和反归一化的操作
class NormalizedActions(gym.ActionWrapper):

    def action(self, action:float):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        # action为tanh的输出，需要转为0～1
        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        #将经过tanh输出的值重新映射回环境的真实值内
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action:float):
        low_bound = self.action_space.low
        upper_bound = self.action_space.high

        #因为激活函数使用的是tanh，这里将环境输出的动作正则化到（-1，1）
        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)
        return action

class DDPG(object):
    def __init__(self, action_dim:int, state_dim:int, hidden_dim:int):
        super(DDPG,self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 128
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 1e-2
        self.replay_buffer_size = 5000
        self.value_lr = 1e-3
        self.policy_lr = 1e-4

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        # 注意：target_value_net为value_net进行滑动平均的参数,只用来存储，不参与训练
        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        # 将 target_value_net 网络的参数初始化为 value_net 的参数
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data) # 原地拷贝

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def ddpg_update(self):
        # state:[batch, state_dim]
        # action:[batch, action_dim]
        # reward:[batch,]
        # next_state:[batch, state_dim]
        # done:[batch,]
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device) # done: 0 or 1

        cur_actor_action = self.policy_net(state)
        critic_loss = self.value_net(state, cur_actor_action) # 当前值的好坏
        critic_loss = -critic_loss.mean()

        # 注意：此处用的是target_xx 来预测action以及value
        next_action = self.target_policy_net(next_state) # actor
        target_next_value = self.target_value_net(next_state, next_action.detach()) # critic
        # 计算Q值, Q(t) = R(t)+gamma*Q(t+1),其中Q(t+1) = ValueFunc(s(t+1), a(t+1))
        expected_cur_value = reward + (1.0 - done) * self.gamma * target_next_value
        expected_cur_value = torch.clamp(expected_cur_value, self.min_value, self.max_value)

        # critic loss
        critic_value = self.value_net(state, action)
        value_loss = self.value_criterion(critic_value, expected_cur_value.detach())

        self.policy_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True) # 注意：不能清空反向传播时的graph, 因为critic反向传播时还会使用value_net的输出
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()  # 此时可以清空反向传播时的graph，与actor_loss一样，他们的梯度也会传到actor网络中去
        self.value_optimizer.step()

        # 将value_net的参数值进行滑动平均后赋给target_value_net, 秒！
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

def plot(frame_idx, rewards):
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def main():
    env:Env = gym.make("Pendulum-v1")
    env = NormalizedActions(env)
    env = gym.wrappers.record_video.RecordVideo(env, f"videos/Pendulumv1/", episode_trigger=lambda epi: epi%1000==0)

    ou_noise = OUNoise(env.action_space)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    ddpg = DDPG(action_dim, state_dim, hidden_dim)

    max_frames = 1000_0000
    max_steps = 5000
    frame_idx = 0
    rewards = []
    batch_size = 1000

    while frame_idx < max_frames:
        state, _ = env.reset()
        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            env.render()
            action = ddpg.policy_net.get_action(state)
            action = ou_noise.get_action(action, step)
            next_state, reward, done, is_truncated, _ = env.step(action)

            ddpg.replay_buffer.push(state, action, reward, next_state, done)
            if len(ddpg.replay_buffer) > batch_size:
                ddpg.ddpg_update()
                ddpg.replay_buffer.clear()

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if frame_idx % max(1000, max_steps + 1) == 0:
                print('frame %s. reward: %s' % (frame_idx, rewards[-1]))
                #plot(frame_idx, rewards)

            if done:
                break

        rewards.append(episode_reward)

    plot(frame_idx, rewards)
    env.close()

if __name__ == '__main__':
    main()