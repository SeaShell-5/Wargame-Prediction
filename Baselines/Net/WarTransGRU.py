import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable
from Baselines.Net.GNN import gtnet
from Baselines.Net.Transnet import EncoderLayer, MultiVariateMultiStepTransformer



class WarTransGRU(torch.nn.Module):
    def __init__(
            self,
            inplanes,
            midplanes1,
            midplanes2,
            outplanes,
            in_vector,
            out_vector,
            in_rnn,
            out_rnn,
            outdense,
            Trans_module,
            GRU_module,
            input_steps,
            out_steps,
            node_num=12,
            num_heads=8,
            en_hidden=1024,
            de_hidden=512
    ):
        super(WarTransGRU, self).__init__()
        self.midplanes1 = midplanes1
        self.midplanes2 = midplanes2
        self.outplanes = outplanes
        self.out_vector = out_vector
        self.plane_length = outdense  # 地图的长 * 宽
        self.out_rnn = out_rnn
        self.inplanes = inplanes
        # self.node_num = node_num
        self.h = None
        self.Trans_module = Trans_module
        self.GRU_module = GRU_module
        

        self.conv0 = nn.Conv2d(in_channels=inplanes, out_channels=midplanes1, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(midplanes1)

        self.conv1 = nn.Conv2d(in_channels=midplanes1, out_channels=midplanes2, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(midplanes2)

        self.conv2 = nn.Conv2d(in_channels=midplanes2, out_channels=outplanes, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.linear_g = nn.Linear(in_vector, out_vector)  

        self.Trans = MultiVariateMultiStepTransformer(outplanes * outdense + out_vector, en_hidden, de_hidden, 1, num_heads, input_steps, out_steps,3)

        # 
        for num in range(3):
            setattr(self, "pre_fc{}".format(num), nn.Linear(de_hidden, in_rnn))
            setattr(self, "rnn{}".format(num), nn.GRUCell(input_size=in_rnn, hidden_size=out_rnn))
            setattr(self, "end_fc{}".format(num), nn.Linear(out_rnn, 1))
        

    def forward(self, states_G, states_S, require_init):
        batch = states_S.size(1)
        seq_len = states_S.size(0)
        self.h = [None for _ in range(3)]
        for num in range(3):
            if self.h[num] is None:
                self.h[num] = Variable(states_S.data.new().resize_((batch, self.out_rnn)).zero_())
            elif True in require_init:
                for idx, init in enumerate(require_init):
                    if init:
                        for num in range(3):
                            self.h[num][idx, :].zero_()
            else:
                pass

        
        values = []
        
        # states_S.shape:torch.Size([T, B, N, map_size[0], map_size[1]])
        x_S = states_S.view(seq_len, batch, -1, states_S.shape[3] * states_S.shape[4])  # x_S1 torch.Size([T, B, N, D])
        x_S = F.relu(self.bn0(self.conv0(x_S.permute(1, 2, 3, 0))))  # x_S2 torch.Size([B, N, F, T])
        x_S = F.relu(self.bn1(self.conv1(x_S)))  
        x_S = F.relu(self.bn2(self.conv2(x_S)))  
        x_S = x_S.view(batch, self.plane_length * self.outplanes, seq_len)
        x_S = x_S.permute(2,0,1)

        x_G = F.relu(self.linear_g(states_G.transpose(0, 1)))
        x_G = x_G.transpose(0, 1)   
        X = torch.cat((x_S, x_G), 2)
        
        X_trans = self.Trans(X.transpose(0, 1)).transpose(0, 1) # x2:torch:[T, B, N]
        
        for idx, (x) in enumerate(X_trans):
            # print(x.shape)
            x_list = [F.relu(getattr(self, "pre_fc{}".format(num))(x)) for num in range(3)]
            self.h = [getattr(self, "rnn{}".format(num))(x_list[num], self.h[num]) for num in range(3)]
            x_list = [torch.sigmoid(getattr(self, "end_fc{}".format(num))(self.h[num])) for num in range(3)]
            value = torch.cat(x_list, dim=1)
            
            values.append(value)
        values = torch.stack(values, dim=0)
        
        return values
        
    
    def detach(self):
        for num in range(3):
            if self.h[num] is not None:
                self.h[num].detach_()






