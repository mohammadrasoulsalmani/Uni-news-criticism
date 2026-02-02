import string
import re
import pandas as pd
import os
import tqdm
from labelling_functions import PS_LF, DRT_LF, PolFol_LF, TweetLevelLF
from feature_extractors import ExtractUserFeats
import re


def check_direct_news_source_replies(text,twitter_handles):
    """
    * Checks if first mention in the tweet is a news source mention
    * Splits on this, and checks the 2nd part contains a news source in the beginning
    """
    reg = r'^@([a-zA-Z0-9_]{1,50})'
    
    # text = x_row.text
    # first split to check if the first mention in the reply chain is a news twitter handle
    first_split = re.split(reg,text)
    
    firstmention_is_news_source = False
    
    if len(first_split)<2:
        # No mentions present, this happens if only the referenced text contained the news mentions
        # return False
        return False
    
    for twh in twitter_handles:
        if twh in first_split[1] or twh.lower() in first_split[1]:
            firstmention_is_news_source=True
            break
    
    # Next check if the remaining part of the split contains a news source in the beginning
    # if true then not a direct news source reply
    
    remaining_begings_with_ns = True
    
    if len(re.split(reg,first_split[-1].strip()))==1:
        remaining_begings_with_ns = False
    
    if firstmention_is_news_source==True and remaining_begings_with_ns == False:
        return True
    else:
        return False

def check_news_content_mentions(text,twitter_handles):
    """
    * Splits on the last mention in the reply chain prefix
    * Check if the text content of the tweet contains a news source mention
    """
    mentions_replychain_identifier = r"^(@([a-zA-Z0-9_]{1,50}\s))+"
    # text = x_row.text
    content = re.split(mentions_replychain_identifier,text)[-1]
    verdict=False # No news mention in content of news
    for twh in twitter_handles:
        if '@'+twh in content or '@'+twh.lower() in content:
            verdict=True
            break
    
    return verdict


def check_replied_to(text,twitter_handles):
    """
    Checks if the news source mentioned, (has to be a mention and not url) is in the content
    of the tweet and not in the initial replied_to chain
    
    * it also checks if the replied_to tweet is repliying to a news source directly
    
    * Positive Example:
        @ArianaGrande Why would @CNN @MSNBC report good news? They are only interested in people being attacked. 
        The more violence the more Trump gets blamed. Rather than bringing people together they do the opposite.
        
        @PalmerReport F$ckers!
    
    * Negative Example:
        @LindseyGrahamSC @seanhannity @FoxNews When I think about you, the word “pathetic”, is flashing across your face like a neon sign. 
        Your time is up. Sit down
        
    Not looking at the referenced text for this
    * as the user is replying and might have a different stance towards the news source the original author is tweeting
    about
    """
    
    if check_direct_news_source_replies(text,twitter_handles):
        # print(f"Direct News Source Reply : True")
        return True
    
    elif check_news_content_mentions(text,twitter_handles):
        # print(f"News Content Mention : True")
        return True
    
    else:
        return False

def remove_non_mentions(df):
    """
    """
    
    def check_nm(tweet_row):
        """
        """
        text = tweet_row.text
        
        if tweet_row.tweet_type == "retweeted" and len(tweet_row.referenced_text)>0:
            
            text = tweet_row.referenced_text[0]
        
        if len(tweet_row.matched_mentions)<=0:
            return True
        else:
            mm = tweet_row.matched_mentions
            verd = True
            for m in mm:
                if "@"+m in text or  "@"+m.lower() in text.lower():
                    verd=False
                    break

            return verd
    
    df["remove_nm_tweet"] = df.apply(lambda x: check_nm(x),axis=1)
    
    df = df.loc[df["remove_nm_tweet"]==False].reset_index(drop=True)
    
    return df

def identify_direct_retweets(df,drt_identifier):
    """
    """
    df = drt_identifier.identify(df)
    return df

def identify_long_replychains(df,twitter_handles):
    """
    """
    df["long_chain_replied_to"] = df.apply(lambda x: not check_replied_to(x.text,twitter_handles) if x.tweet_type=="replied_to" else False,axis=1)
    return df

def translation_init(exceptions=['?','!', '@', '#']):
    """
    """
    puncts = string.punctuation
    for e in exceptions:
        puncts = puncts.replace(e,"")
    tbl = str.maketrans(puncts, ' '*len(puncts)) #map punctuation to space
    return tbl

def replace_newline_tabs(text):
    """
    """
    return re.sub(r"\s+", " ", text)
    
def remove_punctuations(text,translation_table):
    """
    """
    return text.translate(translation_table)


def stitch_tweets(tweet_row):
    """
    """
    if tweet_row["tweet_type"] == "retweeted":
        
        if len(tweet_row["referenced_text"])>0 and len(tweet_row["referenced_text"][0])>0:
            
            text = tweet_row["text"].split(":")[0]+ " " + tweet_row["referenced_text"][0]
            
            return text
        else:
            return tweet_row["text"]
    
    elif tweet_row["tweet_type"] == "replied_to" or tweet_row["tweet_type"] == "quoted":
        
        if len(tweet_row["referenced_text"])>0 and len(tweet_row["referenced_text"][0])>0:
            
            text = tweet_row["referenced_text"][0] + " " +  tweet_row["text"]
            
            return text
        else:
            return tweet_row["text"]
        
    else :

        return tweet_row["text"]

def get_user_label(tweet_row):
    """
    """
    labels = [tweet_row.ps_dist_label,
              tweet_row.ps_dist_drt_label,
              tweet_row.pol_fol_label]
    
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    else:
        return -1
        

def get_combined_vote(tweet_row):
    """
    """
    
    labels = [tweet_row.keyword_label,tweet_row.user_label]
    
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    else:
        return -100 # depicts conflicts

def get_union_label(tweet_row):
    """
    how to consider if the sample belongs in the union of the heuristic
    
    Samples in the Union
    * If only one labels the sample
    * if both label and they agree
    
    Samples not in the Union
    * if both label and they disagree
    * if both don't label
    """
    # when one of the heuristics can't label and the other labels
    if (tweet_row["user_label"] != -1 or tweet_row["keyword_label"] != -1) and sum([tweet_row["user_label"],tweet_row["keyword_label"]]) != -2 :
        
        if tweet_row["user_label"]==-1:
            return tweet_row["keyword_label"]
        elif tweet_row["keyword_label"] == -1:
            return tweet_row["user_label"]
        
    # when both label
    if tweet_row["user_label"] != -1 and tweet_row["keyword_label"] != -1:
        
        # they both agree
        if tweet_row["user_label"] == tweet_row["keyword_label"]:
            return tweet_row["keyword_label"]
        
        else:
            # they disagree
            return -1
    
    # both don't label
    if tweet_row["user_label"] == -1 and tweet_row["keyword_label"] == -1:
        return -1


def load_user_df(path):
    """
    """
    
    user = path.split("/")[-1].split(".")[0]
    df_u = pd.read_pickle(path)
    df_u["user"] = [user]*df_u.shape[0]
    df_u = df_u.loc[df_u["matched_partisans"].str.len()>0]

    df_u = df_u.explode("matched_partisans").reset_index(drop=True)
    df_u["matched_partisans"] = df_u["matched_partisans"].astype(int)
    df_u = df_u.loc[df_u["matched_partisans"].isin([-2,-3,2,3])].reset_index(drop=True)
  
    return df_u


class DRIdentifier(object):
    
    def __init__(self,news_df):
        """
        """
        ex_news_df = news_df.explode("Twitter Handle").reset_index(drop=True).explode("URL").reset_index(drop=True)
        ex_news_twh = ex_news_df["Twitter Handle"].tolist()
        self.news_twh = [e for e in ex_news_twh if type(e)==str]
        
        self.match_pattern = re.compile(r"(RT\s{0,}@[a-zA-Z0-9]*)")
        
    def check_drt(self,tweet_text):
        """
        """
        matches = self.match_pattern.search(tweet_text)
        
        if matches != None:
            matches = matches.group(0)
            acc = matches.replace("RT","").replace(":","").replace("@","").strip()

            if acc in self.news_twh:
                return True
            else:
                return False
        else:
            return False
    
    def identify(self,df):
        """
        """
    
        df["direct_retweet"] = df["text"].apply(lambda x: self.check_drt(x))
        
        return df


def load_and_preprocess(skip_data,all_user_paths=[]):
    """
    """
    path = "../../../Data/new_matched_timelines/"
    following_path = "../../../Data/following_accounts/clean/"
    
    if type(skip_data) != pd.DataFrame or len(all_user_paths)==0:
        skip_users = skip_data["user"].tolist()
        has_following_information = [f for f in os.listdir(following_path)]
        all_user_paths = [path+f for f in os.listdir(path) if ".pkl" in f and f.replace(".pkl","") not in skip_users and f in has_following_information]
    
    print(f"\nTotal Number of Users considered for Train : {len(all_user_paths)}")
    
    news_df = pd.read_pickle("../../../Data/all_news_sources/all_news_sources.pkl")
    pol_df = pd.read_csv("../politicians_twitter.csv")
    
    twitter_handles = news_df.explode("Twitter Handle")
    twitter_handles = twitter_handles.loc[twitter_handles['Twitter Handle'].notna()]
    twitter_handles = twitter_handles["Twitter Handle"].tolist()
    
    drf_ident = DRIdentifier(news_df)
    
    
    ps_dstlf = PS_LF(threshold=0.1,min_count=10)
    ps_dst_drtlf = DRT_LF(threshold=0.15,min_count=25)
    pol_lf = PolFol_LF(pol_df=pol_df,threshold=0.05,min_count=5)
    tweet_lf = TweetLevelLF()
    user_feats_extractor = ExtractUserFeats(news_df,pol_df,following_path="../../../Data/following_accounts/clean/")
    
    samples = []
    users_used = 0
    
    for f in tqdm.tqdm(all_user_paths):
        
        df_u = load_user_df(f)
        df_u["tweet_id"] = df_u["tweet_id"].astype(int)
        
        if df_u.shape[0]>=30:

            df_u = identify_direct_retweets(df_u,drf_ident)
            df_u = identify_long_replychains(df_u,twitter_handles)

            df_u = ps_dstlf.label(df_u)
            df_u = ps_dst_drtlf.label(df_u)
            df_u = pol_lf.label(df_u)
            df_u = tweet_lf.label(df_u)
            df_u = user_feats_extractor.extract(df_u)

            df_u = remove_non_mentions(df_u)
            
            df_u = df_u.loc[df_u["long_chain_replied_to"]==False]
            df_u = df_u.loc[df_u["direct_retweet"]==False]
            df_u = df_u.reset_index(drop=True)

            if df_u.shape[0]>0:
                samples.append(df_u)
                users_used+=1
    
    print(f"\nTotal Number of Users considered for Train after preprocessing : {users_used}")
    
    all_samples = pd.concat(samples).reset_index(drop=True)
    print(f"\nProcessed Data Shape before dropping Duplicates : {all_samples.shape}")
    all_samples = all_samples.drop_duplicates(subset=["tweet_id"],keep="first")
    print(f"\nProcessed Data Shape after dropping Duplicates : {all_samples.shape}")
    
    all_samples = all_samples.reset_index(drop=True)
    
    all_samples["stitched_text"] = all_samples.apply(lambda x: stitch_tweets(x),axis=1)
    
    all_samples["user_label"] =  all_samples.apply(lambda x : get_user_label(x),axis=1)
    
    all_samples["unanimous_label"] = all_samples.apply(lambda x : get_combined_vote(x),axis=1)
    
    all_samples["union_heuristic"] = all_samples.apply(lambda x: get_union_label(x),axis=1)
    
    return all_samples