from Baselines.Net.layer import *


class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device='cuda:0', predefined_A=None,
                 dropout=0.3, dilation_exponential=1, conv_channels=1024, residual_channels=3366, 
                 seq_length=15, in_dim=3366, layers=1, propalpha=0.05, layer_norm_affline=True):
        super(gtnet, self).__init__()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.dy_gconv = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels,
                                    kernel_size=(1, 1))

        self.seq_length = seq_length
        self.layers = layers

        for i in range(1):
            new_dilation = 1
            for j in range(1, layers + 1):

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                if self.gcn_true:
                    self.dy_gconv.append(dy_mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                self.norm.append(LayerNorm((residual_channels, num_nodes, seq_length), elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, idx=None):
        
        # 检查输入
        if torch.isnan(input).any() or torch.isinf(input).any():
            raise ValueError("GNN input contains nan or inf values")
        
        for i in range(self.layers):
            x = nn.functional.pad(input, (6, 0, 0, 0))
            residual = x
            filter = self.filter_convs[i](x)  
            filter = torch.tanh(filter)  
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            
            # 检查卷积输出
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError(f"GNN conv output contains nan or inf values at layer {i}")
            
            if self.gcn_true:
                x = self.dy_gconv[i](x)  # 这里为动态图的图卷积GC_module模块
                # 检查GNN输出
                if torch.isnan(x).any() or torch.isinf(x).any():
                    raise ValueError(f"GNN dy_gconv output contains nan or inf values at layer {i}")
            else:
                x = self.residual_convs[i](x)  
                # 检查残差连接输出
                if torch.isnan(x).any() or torch.isinf(x).any():
                    raise ValueError(f"GNN residual output contains nan or inf values at layer {i}")

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)
            
            # 检查归一化输出
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise ValueError(f"GNN norm output contains nan or inf values at layer {i}")
            
        return x
