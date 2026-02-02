from transformers import AutoModel
import torch.nn as nn
import torch

class UserNetwork(nn.Module):
    """
    3 layer network
    """
    
    def __init__(self,input_shape=2799,l1_h=1024,dropout=0.1,activation="relu"):
        """
        """
        super(UserNetwork, self).__init__()
        
        self.params_ = {"input_shape":input_shape,
                       "l1_h":l1_h,
                        "dropout":dropout,
                        "activation":activation}
        
        l2_h = int(l1_h/2)
        l3_h = int(l2_h/2)
        
        self.fc1 = nn.Linear(input_shape,l1_h)
        # self.fc2 = nn.Linear(l1_h,l2_h)
        self.fc4 = nn.Linear(l1_h,1)
        
        self.dropout = nn.Dropout(dropout)
        
        self.activation = torch.relu
        
        if activation == "relu":
            self.activation = torch.relu
        
        if activation == "sigmoid":
            self.activation = torch.sigmoid
        
    
    def forward(self,x):
        """
        """
        x_1 = self.activation(self.fc1(x))
        x_1 = self.dropout(x_1)
        # x_2 = self.activation(self.fc2(x_1))
        # x_2 = self.dropout(x_2)
        x_4 = torch.sigmoid(self.fc4(x_1))
        return x_1, x_4 
    

class TextNetwork(nn.Module):
    """
    Uses BertTweet
    """
    
    def __init__(self,input_shape=50,lh=128,freeze_bert=True,dropout=0.1,pre_trained="cardiff",pooling="mean",activation="relu"):
        """
        """
        super(TextNetwork, self).__init__()
        
        self.params_ = {"input_shape":input_shape,
                       "lh":lh,
                        "dropout":dropout,
                        "activation":activation,
                        "freeze_bert":freeze_bert,
                        "pre_trained":pre_trained,
                        "pooling":pooling}
        
        self.bert_tweet = None
        self.dropout = nn.Dropout(dropout)
        
        if pre_trained == "vinai":
        
            self.bert_tweet = AutoModel.from_pretrained("vinai/bertweet-base")
        
        if pre_trained == "cardiff":
        
            self.bert_tweet = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        
        
        if freeze_bert:
            for param in self.bert_tweet.parameters():
                param.requires_grad = False
        
        self.pooling = pooling
        
        self.fc_in = nn.Linear(768,lh)
        
        if self.pooling == "concat":
            self.fc_in = nn.Linear(768*2,lh)
            
        self.fc_out = nn.Linear(lh,1)
        
        self.activation = torch.relu
        
        if activation == "relu":
            self.activation = torch.relu
        
        if activation == "sigmoid":
            self.activation = torch.sigmoid
    
    def forward(self,x):
        """
        """
        embeds = self.bert_tweet(x) # returns batch_size x max_length x 768
        
        # pool across max_length
        
        if self.pooling == "mean":
            embeds = torch.mean(embeds.last_hidden_state,axis=1) # batch_size x 768
        
        if self.pooling == "max":
            embeds, max_indices  = torch.max(embeds.last_hidden_state,axis=1)
        
        if self.pooling == "concat":
            
            mean_ = torch.mean(embeds.last_hidden_state,axis=1)
            max_,max_indices = torch.max(embeds.last_hidden_state,axis=1)
            
            embeds = torch.concat((max_,mean_),axis=1)
        
        if self.pooling == "special":
            
            embeds = embeds.last_hidden_state[:,0,:]
        
        x_1 = self.activation(self.fc_in(embeds))
        x_1 = self.dropout(x_1)
        x_2 = torch.sigmoid(self.fc_out(x_1))
        
        return x_1, x_2 
    
class CombinedNetwork(nn.Module):

    
    """
    This network uses user and text features toghether
    """
    def __init__(self,user_input_shape=26,text_lh=128,user_lh=64,freeze_bert=True,dropout=0.1,pre_trained="cardiff",pooling="mean",user_activation="relu",text_activation="sigmoid"):
        """
        """
        super(CombinedNetwork, self).__init__()
        
        self.params_ = {"user_input_shape":user_input_shape,
                       "text_lh":text_lh,
                        "dropout":dropout,
                        "user_lh":user_lh,
                        "freeze_bert":freeze_bert,
                        "pre_trained":pre_trained,
                        "pooling":pooling,
                        "user_activation":user_activation,"text_activation":text_activation}
        
        self.bert_tweet = None
        self.dropout = nn.Dropout(dropout)
        
        if pre_trained == "vinai":
        
            self.bert_tweet = AutoModel.from_pretrained("vinai/bertweet-base")
        
        if pre_trained == "cardiff":
        
            self.bert_tweet = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base")
        
        
        if freeze_bert:
            for param in self.bert_tweet.parameters():
                param.requires_grad = False
        
        self.pooling = pooling
        
        self.fc_text_in = nn.Linear(768,text_lh)
        
        if self.pooling == "concat":
            self.fc_text_in = nn.Linear(768*2,text_lh)
        
        self.text_activation = torch.relu
        
        if text_activation == "relu":
            self.text_activation = torch.relu
        
        if text_activation == "sigmoid":
            self.text_activation = torch.sigmoid
        
        self.user_activation = torch.relu
        
        if user_activation == "relu":
            self.user_activation = torch.relu
        
        if user_activation == "sigmoid":
            self.user_activation = torch.sigmoid
        
        
        self.fc_user_in = nn.Linear(user_input_shape,user_lh)
        
        self.fc_out = nn.Linear(user_lh+text_lh,1)
        
        
    def forward(self,x_user,x_text):
        """
        """
        # text representations and pooling
        
        text_embeds = self.bert_tweet(x_text)
        
        if self.pooling == "mean":
            pooled_embeds = torch.mean(text_embeds.last_hidden_state,axis=1) # batch_size x 768
        
        if self.pooling == "max":
            pooled_embeds, max_indices  = torch.max(text_embeds.last_hidden_state,axis=1)
        
        if self.pooling == "concat":
            
            mean_ = torch.mean(text_embeds.last_hidden_state,axis=1)
            max_,max_indices = torch.max(text_embeds.last_hidden_state,axis=1)
            
            pooled_embeds = torch.concat((max_,mean_),axis=1)
        
        if self.pooling == "special":
            
            pooled_embeds = text_embeds.last_hidden_state[:,0,:]
        
        text_inter = self.text_activation(self.fc_text_in(pooled_embeds))
        
        text_inter = self.dropout(text_inter)
        
        user_inter = self.user_activation(self.fc_user_in(x_user))
        
        user_inter = self.dropout(user_inter)
        
        concat_rep = torch.concat((user_inter,text_inter),axis=1)
        
        out = self.fc_out(concat_rep)
        
        out = torch.sigmoid(out) 
        
        return concat_rep, out