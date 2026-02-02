from preprocessing_utils import check_direct_news_source_replies
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from unidecode import unidecode
from emoji import demojize
from nltk.tokenize import TweetTokenizer

class UserDataset(Dataset):
    """
    Features Included :
    
    # user profile based : DONE
    ----------------------
    * Following vector - 1hot encoded
    * Partisan Dist
        * engaged news sources
        * followed news sources
        * followed politician accounts
    * Frequency of news source mentions
    
    # tweet based :
    ---------------
    * Partisan stance of current tweet - Done
    
    * News source engaged in current tweet - 1hot encoded - Done
    
    * tweet type of current tweet - Done
    
    * is_direct_replied_to news source - Done
    
    * mentions news source in content of the tweet - Done
    
    * is the engaged type : - Done
        * mention
        * url
        * both
    
    * fraction of news mentions that are of the news source in the current tweet - Done
    
    * Num of news sources mentioned - Done
    
    * public metrics :
        * vec of 4 vals
        * {'retweet_count': 0, 'reply_count': 1, 'like_count': 7, 'quote_count': 0}
    
    """
    def __init__(self,dataset_df,news_df,pol_df,label_col="keyword_label"):
        """
        """
        self.df = dataset_df
        
        self.label_col = label_col
        
        self.news_df = news_df
        self.pol_df = pol_df
        
        self.pol_usernames_init()
        self.twitter_handles_init()
        
        self.all_accounts = self.news_twh + self.pol_usernames
        
        self.size = self.df.shape[0]
        
    
    def __len__(self):
        """
        """
        return self.size
    
    def pol_usernames_init(self):
        """
        """
        pol_df = self.pol_df.loc[self.pol_df["party_name"].isin(["Republican","Democrat"])].reset_index(drop=True)
        pol_usernames_tup = list(pol_df[["user_name","party_name"]].itertuples(index=False, name=None))
        self.pol_usernames = [(twh[0],1)  if twh[1]=="Republican" else (twh[0],0) for twh in pol_usernames_tup]
        
    def twitter_handles_init(self):
        """
        """
        twitter_handles = self.news_df.explode("Twitter Handle")
        twitter_handles = twitter_handles.loc[twitter_handles['Twitter Handle'].notna()]
        twitter_handles = list(twitter_handles[["Twitter Handle","Partisan Score"]].itertuples(index=False, name=None))
        twitter_handles_tups = [(twh[0],1)  if twh[1]>0 else (twh[0],0) for twh in twitter_handles]
        self.news_twh = twitter_handles_tups
        self.twh_only = [tup[0] for tup in self.news_twh]
        
    
    def __encode_following__(self,row):
        """
        """
        following = row.accounts_followed
        
        one_hot = [0]*len(self.all_accounts)
        
        for f_idx,a in enumerate(self.all_accounts):
            if a[0] in following:
                one_hot[f_idx]=1
        
        return one_hot
    
    def __encode_partisan_dist__(self,row):
        """
        """
        dist = row.partisan_dist
        
        partisan_vec = [0.0,0.0]
        partisan_vec[0] = dist["cons"]
        partisan_vec[1] = dist["lib"]

        return partisan_vec
        
    
    def __encode_pol_dist__(self,row):
        """
        """
        pol_dist = row.pol_dist

        partisan_vec = [0.0,0.0]
        partisan_vec[0] = pol_dist["cons"]
        partisan_vec[1] = pol_dist["lib"]

        return partisan_vec
    
    def __encode_ns_dist__(self,row):
        """
        """
        following = row.accounts_followed
        news_dist = row.ns_dist

        partisan_vec = [0.0,0.0]
        partisan_vec[0] = news_dist["cons"]
        partisan_vec[1] = news_dist["lib"]

        return partisan_vec
        
    
    def __encode_tweet_stance__(self,row):
        """
        """
        tweet_stance = row.matched_partisans
            
        stances = [-3,-2,2,3]
        stance_vec = [0]*4
        
        for idx,s in enumerate(stances):
            if s == tweet_stance:
                stance_vec[idx] = 1
        
        return stance_vec
    
    def __encode_tweet_type__(self,row):
        """
        """
        tweet_types = ["retweeted","replied_to","status","quoted"]
        
        type_vec = [0]*4
        
        tweet_type = row.tweet_type
        
        for idx,tt in enumerate(tweet_types):
            if tt == tweet_type:
                type_vec[idx] = 1
        
        return type_vec
    
    def __encode_news_source_tweet__(self,row):
        """
        """
        engaged_ns = row.matched_sources
        
        ns_vec = [0]*len(self.news_df["Source"].tolist())
        
        for idx,s in enumerate(self.news_df["Source"].tolist()):
            for n in engaged_ns:
                if s == n:
                    ns_vec[idx] = 1
        
        return ns_vec
                    
    def __encode_engagement_type__(self,row):
        """
        """
        engagement_vec = [0]*3
        matched_mentions = row.matched_mentions
        matched_urls = row.matched_urls
        
        if len(matched_mentions)>0 and len(matched_urls)>0:
            engagement_vec[2]=1
        elif len(matched_mentions)>0 and len(matched_urls)<=0:
            engagement_vec[0]=1
        elif len(matched_mentions)<=0 and len(matched_urls)>0:
            engagement_vec[1]=0
        else:
            raise Exception('Both matched mentions and urls are empty - Not possible')
        
        return engagement_vec
        
    
    def __encode_direct_ns_reply__(self,row):
        """
        """
        direct_reply_vec = [0]
        
        verdict = check_direct_news_source_replies(row.text,self.twh_only)
        
        if verdict:
            direct_reply_vec[0]=1
        else:
            direct_reply_vec[0]=0
        
        return direct_reply_vec
    
    # def __encode_replied_to_content_check__(self,row):
    #     """
    #     """
    #     mention_in_content_vec = [0]
    #     verdict = check_news_content_mentions(row.text,self.twh_only)
        
    #     if verdict:
    #         mention_in_content_vec[0]=1
    #     else:
    #         mention_in_content_vec[0]=0
        
    #     return mention_in_content_vec
    
    def __encode_ns_frac_tweet__(self,row):
        """
        single value representing sum of fraction of news sources engaged in current tweet
        
        eg: engaged with CNN and MSNBC in the current tweet
        
        let K be the total number of news engagements for the user
        
        in the given's user profile:
        * they engaged with CNN total of 10 times 
        * they engaged with MSNBC total of 20 times
        
        * So we return 10/K + 20/K
        
        """
        ns_freq = row.news_source_frequency
        
        total_engagement = sum([ns_freq[ns] for ns in ns_freq])
        
        matched_sources = row.matched_sources
        
        engagement_fracs = []
        
        for m in matched_sources:
            
            assert ns_freq[m]>0
            
            engagement_fracs.append(ns_freq[m]/total_engagement)
        
        summed_frac = sum(engagement_fracs)
        
        return [summed_frac]
    
    def __encode_multiple_ns_mentions__(self,row):
        """
        """
        return [len(row.matched_sources)]
    
    def __encode_tweet_public_metrics__(self,row):
        """
        {'retweet_count': 0, 'reply_count': 1, 'like_count': 7, 'quote_count': 0}
        """
        pm_dic = row.tweet_public_metrics
        
        pms = ["retweet_count","reply_count","like_count","quote_count"]
        
        pm_vec = [0]*4
        
        for idx,pm in enumerate(pms):
            
            pm_vec[idx] = pm_dic[pm]
        
        return pm_vec
    
    def __encode_tweet_direct_rtw_dist__(self,row):
        """
        """
        drt_dist = row.drt_partisan_dist

        partisan_vec = [0.0,0.0]
        partisan_vec[0] = drt_dist["cons"]
        partisan_vec[1] = drt_dist["lib"]

        return partisan_vec
    
    def __getitem__(self,idx):
        """
        """
        row = self.df.iloc[idx,:]
        
        # print(idx)
        
        # user following vector (news source accounts and politicians)
        
        following_vec = self.__encode_following__(row)
        
        # fraction of news partisan mentioned (liberal vs conservative) bin these
        
        partisan_dist = self.__encode_partisan_dist__(row)
        
        # print(partisan_dist)
        
        pol_partisan_dist = self.__encode_pol_dist__(row)
        
        # print(pol_partisan_dist)
        
        ns_partisan_dist = self.__encode_ns_dist__(row)
        
        # print(ns_partisan_dist)
        
        # stance of news source engaged in current tweet
        
        ns_stance = self.__encode_tweet_stance__(row)
              
        tt_vec = self.__encode_tweet_type__(row)

        # ns_vec = self.__encode_news_source_tweet__(row)

        et_vec = self.__encode_engagement_type__(row)

        drns_vec = self.__encode_direct_ns_reply__(row)

        # men_cont_vec = self.__encode_replied_to_content_check__(row)

        num_news_mentions_vec = self.__encode_multiple_ns_mentions__(row)

        twwt_pm_vec = self.__encode_tweet_public_metrics__(row)

        eng_frac = self.__encode_ns_frac_tweet__(row)

        drt_tweet_vec = self.__encode_tweet_direct_rtw_dist__(row)

        feat_vec = following_vec + partisan_dist + pol_partisan_dist + ns_partisan_dist + ns_stance + tt_vec  + et_vec + drns_vec + num_news_mentions_vec +twwt_pm_vec+ eng_frac  + drt_tweet_vec
        
        feat_vec = torch.FloatTensor(feat_vec)
        
        label = torch.Tensor([row[self.label_col].item()])
        
        return {"feats":feat_vec, "label":label , "tweet_id": row.tweet_id, "idx":idx}
        

class TextPreprocessor(object):
    """
    """
    def __init__(self,model_type="cardiff"):
        """
        """
        self.mtype = model_type
        self.tokenizer = TweetTokenizer()
        
    def normalizeToken(self,token):
        """
        """
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return token
    
    def normalize_tweet(self,text):
        """
        """
        tokens = self.tokenizer.tokenize(text.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        return " ".join(normTweet.split())
    
    def cardiff_preprocessor(self,text):
        """
        """
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def vinai_preprocessor(self,text):
        """
        """
        return self.normalize_tweet(text)
    
    def preprocess(self,text):
        """
        """
        if self.mtype == "cardiff":
            return self.cardiff_preprocessor(text)
        else:
            # vinai
            return self.vinai_preprocessor(text)



class TextDataset(Dataset):
    """
    """
    def __init__(self,dataset_df,label_col="keyword_label",pre_trained="cardiff",max_length=50):
        """
        """
        self.df = dataset_df
        
        self.label_col = label_col
        
        self.tokenizer = None
        
        self.pretrained = pre_trained
        
        if pre_trained == "vinai":
        
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        
        if pre_trained == "cardiff":
            
            self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
        
        self.size = self.df.shape[0]
        
        self.preprocessor = TextPreprocessor(model_type=pre_trained)
        
        self.max_length=max_length
    
    def __len__(self):
        """
        """
        return self.size
    
    def __getitem__(self,idx):
        """
        """
        row = self.df.iloc[idx,:]
        
        tweet_text = row.stitched_text
        tweet_text = unidecode(tweet_text)
        
        tweet_text = self.preprocessor.preprocess(tweet_text)
        
        input_ids = torch.tensor(self.tokenizer.encode(tweet_text,padding="max_length", max_length=self.max_length, truncation=True))
        
        label = torch.Tensor([row[self.label_col].item()])
        
        return {"feats":input_ids, "label":label, "idx":idx, "tweet_id":row.tweet_id}
    
class CombinedFeaturesDataset(Dataset):
    """
    returns user_feats,text_feats,label
    """
    def __init__(self,dataset,news_df,pol_df,label_column,max_length=50,pre_trained="cardiff",):
        """
        """
        self.user_dataset = None
        self.text_dataset = None
        
        self.data = dataset
        self.news_df = news_df
        self.pol_df = pol_df
        self.label_column = label_column
        
        self.pre_trained = pre_trained
        self.max_length = max_length
        
    
    
        self.__userdataset_init__()
        self.__textdataset_init__()
    
    
    def __userdataset_init__(self):
        """
        """
        self.user_dataset = UserDataset(dataset_df=self.data,
                                        news_df=self.news_df,
                                        pol_df = self.pol_df,
                                        label_col=self.label_column)
    
    def __textdataset_init__(self):
        """
        """
        self.text_dataset = TextDataset(self.data,
                                 label_col=self.label_column,
                                 pre_trained=self.pre_trained,
                                 max_length=self.max_length)
    
    def  __len__(self):
        """
        """
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        """
        """
        
        try :
            user_sample = self.user_dataset.__getitem__(idx)
            text_sample = self.text_dataset.__getitem__(idx)

            res_ = {}

            res_["user_feat"] = user_sample["feats"]
            res_["text_feat"] = text_sample["feats"]

            assert user_sample["tweet_id"] == text_sample["tweet_id"]

            assert user_sample["label"] == text_sample["label"]

            res_["tweet_id"] = user_sample["tweet_id"]
            res_["idx"] = idx
            res_["label"] = user_sample["label"]

            return res_
        
        except AssertionError:
            print(f"user - {user_sample['tweet_id']}  |  {user_sample['label']}  |  {user_sample['idx']}")
            print(f"text - {text_sample['tweet_id']}  |  {text_sample['label']}  |  {text_sample['idx']}")
            
            raise Exception('Assertion Error - UnifiedDataset')