import torch
import torch.nn as nn

from models.feature_extraction import spec_frontend_CNN, LIPNET_CNN
from models.attention import Attention_Block_defAttentionSequSize



class Fusion_crossmodality_defAttentionSequSize(nn.Module):

    def __init__(self, inpDim: int, procDim: int, outDim: int, blocksize: int, numlayer: int = 8, heads: int = 8, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()

        assert blocksize%4 == 0, "Blocksize must be divisible by 4"

        self.inpDim = inpDim
        self.procDim = procDim
        self.outDim = outDim
        self.blocksize = blocksize
        self.num_layers = numlayer
        self.heads = heads
        self.d_model = procDim
        self.dropout_prob = dropout_prob
        self.bias = bias

        self.value_proj = nn.Linear(self.inpDim, self.procDim)
        self.query_proj = nn.Linear(self.inpDim, self.procDim)

        self.attn_layers = nn.ModuleList([Attention_Block_defAttentionSequSize(blocksize=self.blocksize, heads=self.heads, d_model=self.d_model, dropout_prob=self.dropout_prob, bias=self.bias) for _ in range(self.num_layers)])

        self.out_proj = nn.Linear(self.procDim, self.outDim)


    def forward(self, value, query):

        value = self.value_proj(value)
        query = self.query_proj(query)

        for layer in self.attn_layers:
            value = layer(query=query, key=value, value=value)
            # value = value+value_res
        
        x = self.out_proj(value)
        
        return x


class AV_Fusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.fusion_attLayer = config['model']['AV_Fusion_attLayer']
        self.fusion_attHeads = config['model']['AV_Fusion_attHeads']
        self.fusion_blocksize = config['model']['AV_Fusion_blocksize']
        self.fusion_inp_dim = config['model']['AV_Fusion_inp_dim']
        self.fusion_proc_dim = config['model']['AV_Fusion_proc_dim']
        self.fusion_out_dim = config['model']['AV_Fusion_out_dim']

        self.specfront_procChannels = config['model']['Specfront_procChannels']
        self.specfront_layerNum = config['model']['Specfront_layerNum']
        self.specfront_inp_dim = config['model']['Specfront_inp_dim']
        self.specfront_out_dim = config['model']['Specfront_out_dim']

        self.lipnet_frontend_layers = config['model']['Lipnet_video_layer']
        self.lipnet_emb_size = config['model']['Lipnet_emb_size']

        assert self.fusion_blocksize%4 == 0, "Blocksize must be divisible by 4"

        self.spec_frontend  = spec_frontend_CNN(processing_channels=self.specfront_procChannels, layerNum=self.specfront_layerNum, inp_dim=self.specfront_inp_dim, out_dim=self.specfront_out_dim)
        self.video_frontend = LIPNET_CNN(layers=self.lipnet_frontend_layers, emb_size=self.lipnet_emb_size)
        self.fusion_block = Fusion_crossmodality_defAttentionSequSize(blocksize=self.fusion_blocksize, inpDim=self.fusion_inp_dim, procDim=self.fusion_proc_dim, outDim=self.fusion_out_dim, numlayer=self.fusion_attLayer, heads=self.fusion_attHeads)



    def forward(self, inp_spec, inp_video):

        spec_features  = self.spec_frontend(inp_spec)
        video_features = self.video_frontend(inp_video)

        # Padding to satisfy necessary block size
        sequ_len = spec_features.size()[1]
        if sequ_len%self.fusion_blocksize!=0:
            padding_size = self.fusion_blocksize - sequ_len%self.fusion_blocksize
            spec_features  = torch.nn.functional.pad(spec_features,  (0,0,0,padding_size), "constant", 0)
            video_features = torch.nn.functional.pad(video_features, (0,0,0,padding_size), "constant", 0)
        else:
            padding_size = 0


        x = self.fusion_block(spec_features, video_features)

        # Remove paddings
        x = x[:,:sequ_len,:]
        spec_features  = spec_features[:,:sequ_len,:]
        video_features = video_features[:,:sequ_len,:]

        return x, spec_features, video_features
