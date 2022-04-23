from itertools import chain

import spinup.algos.pytorch.ppo.core as core
import torch
from torch import nn


class TransferNet(nn.Module):

    def __init__(self, net, pre, post=None):
        super().__init__()
        self.net = net # Value or Mu MLP
        self.pre = pre
        self.post = post

    def forward(self, obs):
        obs = obs + self.pre(obs)
        outputs = self.net(obs)
        if hasattr(self, 'post') and self.post is not None:
            outputs = outputs + self.post(outputs)
        return outputs


def add_layers(ac, num_pre_layers=0, num_post_layers=0, hidden_size=64):
    # Observation preprocessing
    in_features = ac.pi.mu_net[0].in_features
    in_sizes = [in_features] + [hidden_size]*(num_pre_layers-1) + [in_features]
    # Action postprocessing
    out_features = ac.pi.mu_net[-2].out_features
    out_sizes = [out_features] + [hidden_size]*(num_post_layers-1) + [out_features]

    pre_pi = core.mlp(in_sizes, nn.Tanh)
    pre_v = core.mlp(in_sizes, nn.Tanh)
    post = core.mlp(out_sizes, nn.Tanh)

    for net in [pre_pi, pre_v, post]:
        for p in net.parameters():
            p.data *= 0.01
            # p.data.fill_(0)

    pi_params = chain(pre_pi.parameters(), post.parameters())
    v_params = pre_v.parameters()

    pi_tn = TransferNet(ac.pi.mu_net, pre_pi, post=post)
    v_tn = TransferNet(ac.v.v_net, pre_v)
    ac.pi.mu_net = pi_tn
    ac.v.v_net = v_tn

    return ac, (pi_params, v_params)


if __name__ == '__main__':
    # pass
    m = torch.load('models/walker2d_base/model.pt')
    print(m)
    m, _ = add_layers(m, 2, 2, hidden_size=32)
    print(m)
    # torch.save(m, 'models/walker2d(2-0)_base/model.pt')
