import pandas as pd
import numpy as np
import tqdm
import torch
from torch_datasets import TextDataset,CombinedFeaturesDataset, UserDataset
from preprocessing_utils import load_user_df,DRIdentifier,identify_direct_retweets,identify_long_replychains
from feature_extractors import ExtractUserFeats
from preprocessing_utils import remove_non_mentions
import os



def predict_mocking(user_df,device):
    """
    """
    model = torch.load("model_scripts_results/best_models/dawid_skene_text_model.pth")
    
    model.to(device)
    
    model.eval()
    
    
    duser = TextDataset(user_df,
                      label_col="tweet_id",
                      max_length=50,
                      pre_trained="cardiff")
    
    dloader = torch.utils.data.DataLoader(duser,
                                          batch_size=1000,
                                          num_workers=10)
    
    tweet_ids = []
    preds_ = []
    pred_probs = []
    
    with torch.no_grad():
        for t_idx,t_batch in enumerate(dloader):
            
            t_feats = t_batch["feats"]
            tweet_id = t_batch["tweet_id"]
            
            t_feats = t_feats.to(device)
            
            ti,outs_t = model(t_feats)
            
            pp = outs_t.reshape(-1).detach().cpu().numpy()
            preds = outs_t.reshape(-1).detach().cpu().numpy().round()
            
            preds_.append(preds)
            tweet_ids.append(tweet_id)
            pred_probs.append(pp)
        
    
    preds_ = np.concatenate(preds_)
    tweet_ids = np.concatenate(tweet_ids)
    pred_probs = np.concatenate(pred_probs)
    
    df_preds = pd.DataFrame()
    df_preds["tweet_id"] = tweet_ids
    df_preds["model_preds"] = preds_
    df_preds["model_preds_prob"] = pred_probs
    
    df_labelled = user_df.merge(df_preds,on="tweet_id",how="inner")
    
    print(df_labelled.columns)
    
    model = None

    del model

    torch.cuda.empty_cache()
    
    return df_labelled



def get_labels_distrust(users_used_paths):
    """
    """
    path = "../../../Data/new_matched_timelines/"
    fol_path = "../../../Data/following_accounts/clean/"
    
    following = [f for f in os.listdir(fol_path) if ".pkl" in f]
    
    files = [path+f for f in users_used_paths if ".pkl" in f and f in following]
    
    print(len(files))
    
    pol_df = pd.read_csv("../politicians_twitter.csv")
    
    news_df = pd.read_pickle("../../../Data/all_news_sources/all_news_sources.pkl")
    
    drf_ident = DRIdentifier(news_df)
    user_feats_extractor = ExtractUserFeats(news_df,pol_df,following_path="../../../Data/following_accounts/clean/")
    
    twitter_handles = news_df.explode("Twitter Handle")
    twitter_handles = twitter_handles.loc[twitter_handles['Twitter Handle'].notna()]
    twitter_handles = twitter_handles["Twitter Handle"].tolist()
    labelled_samples = []

    for f in tqdm.tqdm(files):
        df_u = load_user_df(f)
        
        if df_u.shape[0]>30:
            
            user = df_u["user"].iloc[0]
            df_u["tweet_id"] = df_u["tweet_id"].astype(np.int64)
            df_u = df_u.drop_duplicates(subset="tweet_id",keep=False)
            df_u = identify_direct_retweets(df_u,drf_ident)
            df_u = identify_long_replychains(df_u,twitter_handles)
            df_u = user_feats_extractor.extract(df_u)
            df_u["stitched_text"] = df_u.apply(lambda x: stitch_tweets(x),axis=1)
            
            labelled_samples.append(df_u)
    
    all_labelled = pd.concat(labelled_samples).reset_index(drop=True)
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    df_u_labelled = predict_mocking(all_labelled,device,news_df,pol_df)
    
    return df_u_labelled

def filter_user_df(user_df,verbose=0):
    """
    Remove tweets that don't mention news source in content of the tweet,
    long reply chains and reverse labels for direct retweets
    """
    user_df = user_df.loc[user_df["matched_sources"].str.len()<2]
    
    user_df.loc[(user_df["model_preds"]==1) & (user_df["direct_retweet"]==True),"model_preds"] = 0
    
    user_df_trusted = user_df.loc[user_df["model_preds"]==0]
    user_df_distrusted = user_df.loc[user_df["model_preds"]==1]
    
    user_df_distrusted = user_df_distrusted.loc[(user_df_distrusted["direct_retweet"]==False) & (user_df_distrusted["long_chain_replied_to"]==False)]
    if verbose == 1:
        print(f"Before removing non-mentions (all distrusted tweets) : {user_df_distrusted.shape}")
    if user_df_distrusted.shape[0]>0:
        user_df_distrusted = remove_non_mentions(user_df_distrusted)
        
        if verbose == 1:
            print(f"After removing non-mentions (all distrusted tweets) : {user_df_distrusted.shape}")
        
    
    user_df = pd.concat([user_df_trusted,user_df_distrusted]).reset_index(drop=True)
    
    return user_df



def predict_text_model(model,data,heu_type,device):
    """
    """
    
    dataset = TextDataset(data,
                          label_col="annotations",
                          pre_trained="cardiff",
                          max_length=50)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1000,
                                             num_workers=10)
    probs_ = []
    preds_ = []
    truths_ = []
    tweet_ids = []
    
    model.eval()
    with torch.no_grad():
        for t_idx,t_batch in enumerate(dataloader):
            feats = t_batch["feats"]
            labels = t_batch["label"]
            tweet_id = t_batch["tweet_id"]
            
            inputs = feats.to(device)
            labels = labels.to(device)

            _,outs = model(inputs)
            
            preds = outs.reshape(-1).detach().cpu().numpy().round()
            
            preds_.append(preds)
            
            truths_.append(labels.data.cpu().numpy())
            
            tweet_ids.append(tweet_id.numpy())
            
            pp = outs.reshape(-1).detach().cpu().numpy()
            
            probs_.append(pp)

    probs_ = np.concatenate(probs_)
    preds_ = np.concatenate(preds_)
    truths_ = np.concatenate(truths_)
    tweet_ids = np.concatenate(tweet_ids)
    
    pred_df = pd.DataFrame()
    pred_df["tweet_id"] = tweet_ids
    pred_df[f"text_model_{heu_type}_pred"] = preds_
    pred_df[f"text_model_{heu_type}_pred_prob"] = probs_
    pred_df[f"text_model_{heu_type}_pred_conf"] = pred_df.apply(lambda x: x[f"text_model_{heu_type}_pred_prob"] if x[f"text_model_{heu_type}_pred"]==1 else 1-x[f"text_model_{heu_type}_pred_prob"],axis=1)
    
    return pred_df
    
    
    

def predict_user_model(model,data,heu_type,device):
    """
    """
    news_df = pd.read_pickle("../../../Data/all_news_sources/all_news_sources.pkl")
    pol_df = pd.read_csv("../politicians_twitter.csv")
    
    dataset = UserDataset(data,
                          news_df = news_df,
                          pol_df = pol_df,
                          label_col="annotations")
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1000,
                                             num_workers=10)
    probs_ = []
    preds_ = []
    truths_ = []
    tweet_ids = []
    
    model.eval()
    with torch.no_grad():
        for t_idx,t_batch in enumerate(dataloader):
            feats = t_batch["feats"]
            labels = t_batch["label"]
            tweet_id = t_batch["tweet_id"]
            
            inputs = feats.to(device)
            labels = labels.to(device)

            _,outs = model(inputs)
            
            preds = outs.reshape(-1).detach().cpu().numpy().round()
            
            preds_.append(preds)
            
            truths_.append(labels.data.cpu().numpy())
            
            tweet_ids.append(tweet_id.numpy())
            
            pp = outs.reshape(-1).detach().cpu().numpy()
            
            probs_.append(pp)

    probs_ = np.concatenate(probs_)
    preds_ = np.concatenate(preds_)
    truths_ = np.concatenate(truths_)
    tweet_ids = np.concatenate(tweet_ids)
    
    pred_df = pd.DataFrame()
    pred_df["tweet_id"] = tweet_ids
    pred_df[f"user_model_{heu_type}_pred"] = preds_
    pred_df[f"user_model_{heu_type}_pred_prob"] = probs_
    pred_df[f"user_model_{heu_type}_pred_conf"] = pred_df.apply(lambda x: x[f"user_model_{heu_type}_pred_prob"] if x[f"user_model_{heu_type}_pred"]==1 else 1-x[f"user_model_{heu_type}_pred_prob"],axis=1)
    
    return pred_df

def predict_combined_model(model,data,heu_type,device):
    """
    """
    news_df = pd.read_pickle("../../../Data/all_news_sources/all_news_sources.pkl")
    pol_df = pd.read_csv("../politicians_twitter.csv")
    
    dataset = CombinedFeaturesDataset(data, 
                                      news_df,
                                      pol_df,
                                      label_column="annotations",
                                      max_length=50,
                                      pre_trained="cardiff")
    
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1000,
                                             num_workers=10)
    
    preds_ = []
    probs_ = []
    truths_ = []
    tweet_ids = []
    
    model.eval()
    with torch.no_grad():
        for t_idx,t_batch in enumerate(dataloader):
            
            u_feats = t_batch["user_feat"]
            t_feats = t_batch["text_feat"]
            labels = t_batch["label"]
            tweet_id = t_batch["tweet_id"]

            u_feats, t_feats = u_feats.to(device),t_feats.to(device)
            labels = labels.to(device)

            _,outs = model(u_feats,t_feats)
            
            preds = outs.reshape(-1).detach().cpu().numpy().round()
            
            preds_.append(preds)
            
            truths_.append(labels.data.cpu().numpy())
            
            tweet_ids.append(tweet_id.numpy())
            
            pp = outs.reshape(-1).detach().cpu().numpy()
            
            probs_.append(pp)

    probs_ = np.concatenate(probs_)
    preds_ = np.concatenate(preds_)
    truths_ = np.concatenate(truths_)
    tweet_ids = np.concatenate(tweet_ids)
    
    pred_df = pd.DataFrame()
    pred_df["tweet_id"] = tweet_ids
    pred_df[f"combined_model_{heu_type}_pred"] = preds_
    pred_df[f"combined_model_{heu_type}_pred_prob"] = probs_
    pred_df[f"combined_model_{heu_type}_pred_conf"] = pred_df.apply(lambda x: x[f"combined_model_{heu_type}_pred_prob"] if x[f"combined_model_{heu_type}_pred"]==1 else 1-x[f"combined_model_{heu_type}_pred_prob"],axis=1)
    
    
    return pred_df


def predict_labels(dataset,heu_type="user"):
    """
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if heu_type == "user":
        
        text_model = torch.load("model_scripts_results/best_models/text_model_user.pth")
        
        text_model.to(device)
        
        text_preds = predict_text_model(model=text_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        text_model = None

        del text_model

        torch.cuda.empty_cache()
        
        
        user_model = torch.load("model_scripts_results/best_models/user_model_user.pth")
        
        user_model.to(device)
        
        user_preds = predict_user_model(model=user_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        user_model = None

        del user_model

        torch.cuda.empty_cache()
        
        
        combined_model =  torch.load("model_scripts_results/best_models/combined_model_user.pth")
        
        combined_model.to(device)
        
        combined_preds = predict_combined_model(model=combined_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        combined_model = None

        del combined_model

        torch.cuda.empty_cache()
        
        
        combined_model =  torch.load("model_scripts_results/best_models/combined_model_user_self_trained.pth")
        
        combined_model.to(device)
        
        combined_preds_self = predict_combined_model(model=combined_model,
                                        data=dataset,
                                        heu_type="user_self_trained",
                                        device=device)
        
        combined_model = None

        del combined_model

        torch.cuda.empty_cache()
        
        dfs = [df.set_index('tweet_id') for df in [text_preds,user_preds,combined_preds,combined_preds_self]]
        
        pred_df = pd.concat(dfs, axis=1)
    
        return pred_df
    
    if heu_type == "text":
        
        text_model = torch.load("model_scripts_results/best_models/text_model_text.pth")
        
        text_model.to(device)
        
        text_preds = predict_text_model(model=text_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        text_model = None

        del text_model

        torch.cuda.empty_cache()
        
        
        user_model = torch.load("model_scripts_results/best_models/user_model_text.pth")
        
        user_model.to(device)
        
        user_preds = predict_user_model(model=user_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        user_model = None

        del user_model

        torch.cuda.empty_cache()
        
        
        combined_model =  torch.load("model_scripts_results/best_models/combined_model_text.pth")
        
        combined_model.to(device)
        
        combined_preds = predict_combined_model(model=combined_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        combined_model = None

        del combined_model

        torch.cuda.empty_cache()
        
        dfs = [df.set_index('tweet_id') for df in [text_preds,user_preds,combined_preds]]
        
        pred_df = pd.concat(dfs, axis=1)
    
        return pred_df
    
    if heu_type == "union":
        
        text_model = torch.load("model_scripts_results/best_models/text_model_union.pth")
        
        text_model.to(device)
        
        text_preds = predict_text_model(model=text_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        text_model = None

        del text_model

        torch.cuda.empty_cache()
        
        
        user_model = torch.load("model_scripts_results/best_models/user_model_union.pth")
        
        user_model.to(device)
        
        user_preds = predict_user_model(model=user_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        user_model = None

        del user_model

        torch.cuda.empty_cache()
        
        
        combined_model =  torch.load("model_scripts_results/best_models/combined_model_union.pth")
        
        combined_model.to(device)
        
        combined_preds = predict_combined_model(model=combined_model,
                                        data=dataset,
                                        heu_type=heu_type,
                                        device=device)
        
        combined_model = None

        del combined_model

        torch.cuda.empty_cache()
        
        
        combined_model =  torch.load("model_scripts_results/best_models/combined_model_union_self_trained.pth")
        
        combined_model.to(device)
        
        combined_preds_self = predict_combined_model(model=combined_model,
                                        data=dataset,
                                        heu_type="union_self_trained",
                                        device=device)
        
        combined_model = None

        del combined_model

        torch.cuda.empty_cache()
        
        dfs = [df.set_index('tweet_id') for df in [text_preds,user_preds,combined_preds,combined_preds_self]]
        
        pred_df = pd.concat(dfs, axis=1)
    
        return pred_df
    
    