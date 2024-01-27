import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from projection_layer_model import build_resnet_projection_layer



class MultiModalGPT(nn.Module):
    """
    Pytorch Lightning module for Transformer

    """
    def __init__(self,
                 llm_model_name,
                 projection_layer_in_channels,
                 projection_layer_out_channels,
                 projection_layer_depth, 
                 tokenizer,
                 device,
                 ):
        super(MultiModalGPT, self).__init__()
        self.llm_model = None
        self.tokenizer = None
        self.llm_model, self.tokenizer = self._load_pretrained_llm(llm_model_name)
        self.projection_layer = self._build_projection_layer(projection_layer_in_channels,
                                                             projection_layer_out_channels,
                                                             projection_layer_depth)
        self.device = device
    

    def _load_pretrained_llm(self, llm_model_name):
        """
        Load the model, tokenizer 
        """
        model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        for param in model.parameters():
            param.requires_grad = False
        return model, tokenizer


    def _build_projection_layer(self, 
                                projection_layer_in_channels, 
                                projection_layer_out_channels, 
                                projection_layer_depth):
        return build_resnet_projection_layer(projection_layer_in_channels,
                                             projection_layer_out_channels,
                                             projection_layer_depth)

    
    def forward(self, x):
        x = self.projection_layer(x)
        with torch.no_grad():  
            x = self.llm_model(x)
        return x



    



    

