
import numpy as np
import random
import tqdm
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import torch.nn as nn
import torch

from models import UserNetwork, TextNetwork, CombinedNetwork




class EarlyStopping():
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        # self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def set_seed(seed=42):
    """Set seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_network(dataloader,model,device):
    """
    """
    acc = 0.0
    preds_ = []
    truths_ = []
    model.eval()
    with torch.no_grad():
        for t_idx,t_batch in enumerate(dataloader):
            feats = t_batch["feats"]
            labels = t_batch["label"]
            
            inputs = feats.to(device)
            labels = labels.to(device)

            _,outs = model(inputs)
            
            preds = outs.reshape(-1).detach().cpu().numpy().round()
            
            preds_.append(preds)
            
            truths_.append(labels.data.cpu().numpy())

    
    preds_ = np.concatenate(preds_)
    truths_ = np.concatenate(truths_)
    print(classification_report(truths_,preds_,zero_division=1,digits=4))
    print(f"ROC AUC : {roc_auc_score(truths_,preds_) : .4}")
    return (classification_report(truths_,preds_,zero_division=1,digits=4,output_dict=True),roc_auc_score(truths_,preds_))

def evaluate_network(dataloader,model,loss,device,size):
    """
    """
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    preds_ = []
    truths_ = []
    
    model.eval()
    
    with torch.no_grad():
        
        
        for t_idx,t_batch in enumerate(dataloader):
        
            feats = t_batch["feats"]
            labels = t_batch["label"]

            inputs = feats.to(device)
            labels = labels.to(device)

            _,outs = model(inputs)
            
            avg_sample_loss_per_batch = loss(outs,labels)
            
            total_batch_loss = avg_sample_loss_per_batch.item() * labels.shape[0]

            epoch_loss += total_batch_loss

            preds = outs.reshape(-1).detach().cpu().numpy().round()

            correct = accuracy_score(labels.data.cpu().numpy(), preds,normalize=False)

            epoch_acc += correct
            
            preds_.append(preds)
            
            truths_.append(labels.data.cpu().numpy())

        epoch_acc = (epoch_acc / size) * 100
    
    preds_ = np.concatenate(preds_)
    truths_ = np.concatenate(truths_)
    
    scores = classification_report(truths_,preds_,zero_division=1,output_dict=True,digits=4)
    
    f1_wa = scores["weighted avg"]["f1-score"]
    
    return epoch_loss, epoch_acc, f1_wa

def train_network(train_dataset,val_dataset,model,lr,epochs,batch_size,seed=42,plot=False,patience=5,user=True,device_ids=[0, 2, 3]):
    """
    """
    
    set_seed(seed=seed)
    
    def _init_fn(worker_id):
        set_seed(seed+worker_id)
    
    if len(device_ids)>1:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        model.to(device)
    
        if torch.cuda.device_count() > 1:
            print("\nLet's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model,device_ids=device_ids)
    
    else:
        
        device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    
        model.to(device)


    
    print(f"Train Size : {len(train_dataset)}")
    print(f"Val Size : {len(val_dataset)}")

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   num_workers=10,worker_init_fn=_init_fn)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 shuffle=False,
                                                 batch_size=batch_size,
                                                 num_workers=10,worker_init_fn=_init_fn)
    
    optimizer = torch.optim.Adam( model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    
    loss = torch.nn.BCELoss()
    
    # train model
    
    best_models = []
    
    pbar = tqdm.tqdm(range(epochs),desc="Epoch")
    
    early_stopping = EarlyStopping(patience=patience)
    
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accs = []
    epoch_val_accs = []
    
    track_models = []
    
    for e in pbar:
        
        model.train()
        
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_acc = 0.0
        epoch_val_acc = 0.0
    
        for t_idx,t_batch in enumerate(train_dataloader):

            feats = t_batch["feats"]
            labels = t_batch["label"]

            inputs = feats.to(device)
            labels = labels.to(device)

            _,outs = model(inputs)

            optimizer.zero_grad()

            avg_sample_loss_per_batch = loss(outs,labels)

            avg_sample_loss_per_batch.backward()

            optimizer.step()

            total_batch_loss = avg_sample_loss_per_batch.item() * labels.shape[0]

            epoch_train_loss += total_batch_loss
            
            train_pred = outs.reshape(-1).detach().cpu().numpy().round()

            correct = accuracy_score(labels.data.cpu().numpy(), train_pred,normalize=False)

            epoch_train_acc += correct


        epoch_train_acc_ = (epoch_train_acc / len(train_dataset)) * 100
                 
        epoch_val_loss, epoch_val_acc, f1_wa = evaluate_network(dataloader=val_dataloader,model=model,loss=loss,device=device,size=len(val_dataset))
        
        pbar.set_postfix(train_loss=round(epoch_train_loss/len(train_dataset),5),
                         val_loss=round(epoch_val_loss/len(val_dataset),5),
                         train_acc=epoch_train_acc_,
                         val_acc= epoch_val_acc,
                         val_f1_wa = f1_wa)
        
        epoch_train_losses.append(round(epoch_train_loss/len(train_dataset),5))
        epoch_val_losses.append(round(epoch_val_loss/len(val_dataset),5))
        epoch_train_accs.append(epoch_train_acc_)
        epoch_val_accs.append(epoch_val_acc)
        
        early_stopping(np.mean(epoch_val_loss))
        
        state_temp = None
        
        if len(device_ids)>1:
            state_temp = copy.deepcopy(model.module.state_dict())
        else:
            state_temp = copy.deepcopy(model.state_dict())
            
        track_models.append((epoch_val_acc,state_temp))
        
        if early_stopping.early_stop:
            print("Early stopping ....")
            break
        
    
    if plot:
        plot_losses(epoch_train_losses,epoch_val_losses,acc=False)

        plot_losses(epoch_train_accs,epoch_val_accs,acc=True)
    
    sorted_tm = sorted(track_models,key=lambda x: x[0],reverse=True)
    
    best_m = sorted_tm[0][1]
    
    params_b = None
    
    if len(device_ids)>1:
        params_b = model.module.params_
    else:
        params_b = model.params_
        
    if user:
        best_model = UserNetwork(**params_b).to(torch.device("cpu"))
    
    if not user:
        best_model = TextNetwork(**params_b).to(torch.device("cpu"))
        
    best_model.load_state_dict(best_m)
    best_model.eval()
    
    return best_model, epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs
    

def plot_losses(epochs_loss_train,epochs_loss_val,acc=False):
    """
    epochs_loss_train - list
    epochs_loss_val - list
    """
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    epochs = [i for i in range(len(epochs_loss_train))]
    ax.plot(epochs,epochs_loss_train,marker="o",markersize=10,label="Train",color="tab:red",alpha=0.8)
    ax.plot(epochs,epochs_loss_val,marker="o",markersize=10,label="Validation",color="tab:blue",alpha=0.8)
    if not acc:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
    if acc:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
    
    ax.grid(True)
    ax.legend()
    plt.show()
    pass


def test_combined_network(dataloader,model,device):
    """
    """
    acc = 0.0
    preds_ = []
    truths_ = []
    
    model.eval()
    
    with torch.no_grad():
        for t_idx,t_batch in enumerate(dataloader):
            
            u_feats = t_batch["user_feat"]
            t_feats = t_batch["text_feat"]
            labels = t_batch["label"]

            u_feats, t_feats = u_feats.to(device),t_feats.to(device)
            labels = labels.to(device)

            _,outs = model(u_feats,t_feats)
            
            preds = outs.reshape(-1).detach().cpu().numpy().round()
            
            preds_.append(preds)
            
            truths_.append(labels.data.cpu().numpy())

    
    preds_ = np.concatenate(preds_)
    truths_ = np.concatenate(truths_)
    print(classification_report(truths_,preds_,zero_division=1,digits=4))
    print(f"ROC AUC : {roc_auc_score(truths_,preds_) : .4}")
    
    return (classification_report(truths_,preds_,zero_division=1,digits=4,output_dict=True),roc_auc_score(truths_,preds_))

def evaluate_combined_network(dataloader,model,loss,device,size):
    """
    """
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    preds_ = []
    truths_ = []
    
    model.eval()
    
    with torch.no_grad():
        
        
        for t_idx,t_batch in enumerate(dataloader):
        
            u_feats = t_batch["user_feat"]
            t_feats = t_batch["text_feat"]
            labels = t_batch["label"]

            u_feats, t_feats = u_feats.to(device),t_feats.to(device)
            labels = labels.to(device)

            _,outs = model(u_feats,t_feats)
            
            avg_sample_loss_per_batch = loss(outs,labels)
            
            total_batch_loss = avg_sample_loss_per_batch.item() * labels.shape[0]

            epoch_loss += total_batch_loss

            preds = outs.reshape(-1).detach().cpu().numpy().round()

            correct = accuracy_score(labels.data.cpu().numpy(), preds,normalize=False)

            epoch_acc += correct
            
            preds_.append(preds)
            
            truths_.append(labels.data.cpu().numpy())

        epoch_acc = (epoch_acc / size) * 100
    
    preds_ = np.concatenate(preds_)
    truths_ = np.concatenate(truths_)
    
    scores = classification_report(truths_,preds_,zero_division=1,output_dict=True,digits=4)
    
    f1_wa = scores["weighted avg"]["f1-score"]
    
    return epoch_loss, epoch_acc, f1_wa



def train_combined_network(train_dataset,
                             val_dataset,
                             model,
                             lr,
                             epochs,
                             batch_size,
                             seed=42,
                             plot=False,
                             patience=5,
                             debug=False,
                           device_ids=[0,2,3]):
    """
    """
    set_seed(seed=seed)
    
    def _init_fn(worker_id):
        set_seed(seed+worker_id)
    
    
    if len(device_ids)>1:
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        model.to(device)
    
        if torch.cuda.device_count() > 1:
            print("\nLet's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model,device_ids=device_ids)
    
    else:
        
        device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    
        model.to(device)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   shuffle=False,
                                                   batch_size=batch_size,
                                                   num_workers=10,
                                                   worker_init_fn=_init_fn)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 shuffle=False,
                                                 batch_size=batch_size,
                                                 num_workers=10,
                                                 worker_init_fn=_init_fn)
    
    
    print(f"Train Size : {len(train_dataset)}")
    print(f"Val Size : {len(val_dataset)}")
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    
    loss = torch.nn.BCELoss()
    
    # train model
    
    pbar = tqdm.tqdm(range(epochs),desc="Epoch")
    
    early_stopping = EarlyStopping(patience=patience)
    
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accs = []
    epoch_val_accs = []
    
    tups_for_ulpred = []
    track_models = []
    
    
    for e in pbar:
        
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_acc = 0.0
        epoch_val_acc = 0.0
        
        model.train()
    
        for t_idx,t_batch in enumerate(train_dataloader):

            u_feats = t_batch["user_feat"]
            t_feats = t_batch["text_feat"]
            labels = t_batch["label"]

            u_feats, t_feats = u_feats.to(device),t_feats.to(device)
            labels = labels.to(device)

            _,outs = model(u_feats,t_feats)

            optimizer.zero_grad()

            avg_sample_loss_per_batch = loss(outs,labels)

            avg_sample_loss_per_batch.backward()

            optimizer.step()

            total_batch_loss = avg_sample_loss_per_batch.item() * labels.shape[0]

            epoch_train_loss += total_batch_loss
            
            train_pred = outs.reshape(-1).detach().cpu().numpy().round()

            correct = accuracy_score(labels.data.cpu().numpy(), train_pred,normalize=False)

            epoch_train_acc += correct
        
        
        epoch_train_acc_ = (epoch_train_acc / len(train_dataset)) * 100
                 
        epoch_val_loss, epoch_val_acc, f1_wa = evaluate_combined_network(dataloader=val_dataloader,model=model,loss=loss,device=device,size=len(val_dataset))
        
        pbar.set_postfix(train_loss=round(epoch_train_loss/len(train_dataset),5),
                         val_loss=round(epoch_val_loss/len(val_dataset),5),
                         train_acc=epoch_train_acc_,
                         val_acc= epoch_val_acc,
                         val_f1_wa = f1_wa)
        
        
        epoch_train_losses.append(round(epoch_train_loss/len(train_dataset),5))
        epoch_val_losses.append(round(epoch_val_loss/len(val_dataset),5))
        epoch_train_accs.append(epoch_train_acc_)
        epoch_val_accs.append(epoch_val_acc)
        
        early_stopping(np.mean(epoch_val_loss))
        
        state_temp = None
        
        if len(device_ids)>1:
            state_temp = copy.deepcopy(model.module.state_dict())
        else:
            state_temp = copy.deepcopy(model.state_dict())
            
        track_models.append((epoch_val_acc,state_temp))
        
        if early_stopping.early_stop:
            print("Early stopping ....")
            break
        
          
    if plot:
        plot_losses(epoch_train_losses,epoch_val_losses,acc=False)

        plot_losses(epoch_train_accs,epoch_val_accs,acc=True)
    
    sorted_tm = sorted(track_models,key=lambda x: x[0],reverse=True)
    
    best_m = sorted_tm[0][1]
    
    params_b = None
    
    if len(device_ids)>1:
        params_b = model.module.params_
    else:
        params_b = model.params_
    
    
    best_model = CombinedNetwork(**params_b).to(torch.device("cpu"))
    best_model.load_state_dict(best_m)
    best_model.eval()
    
    return best_model, epoch_train_losses, epoch_val_losses, epoch_train_accs, epoch_val_accs