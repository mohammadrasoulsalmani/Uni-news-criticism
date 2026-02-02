import os
import pandas as pd
import re
from preprocessing_utils import translation_init,remove_punctuations
from collections import defaultdict, Counter


class PS_LF(object):
    """
    Heuristic Based on Partisan Stance Distribution
    """
    def __init__(self,threshold,min_count):
        """
        """
        self.threshold=threshold
        self.min_count=min_count
    
    def __get_partisan_dist__(self,df):
        """
        """
        dist = df["matched_partisans"].value_counts().to_dict()
        norm_dist = {d: dist[d]/df.shape[0] for d in dist}
        norm_dist = defaultdict(int,norm_dist)

        conservative_score = norm_dist[float(2)] + norm_dist[float(3)]
        liberal_score = norm_dist[float(-2)] + norm_dist[float(-3)]

        return {"cons":round(conservative_score,4), "libs":round(liberal_score,4)}
    
    def label(self,df):
        """
        """
        
        if df.shape[0]>0:
            
            if df.shape[0]>self.min_count:
        
                ps_dist = self.__get_partisan_dist__(df)

                if ps_dist["cons"]>ps_dist["libs"] and ps_dist["libs"]<=self.threshold:
                    df["ps_dist_label"] = df["matched_partisans"].apply(lambda x: 1 if x<0 else 0)

                elif ps_dist["libs"]>ps_dist["cons"] and ps_dist["cons"]<=self.threshold:
                    df["ps_dist_label"] = df["matched_partisans"].apply(lambda x: 1 if x>0 else 0)
                else:
                    df["ps_dist_label"] = df["matched_partisans"].apply(lambda x: -1)
            else:
                df["ps_dist_label"] = df["matched_partisans"].apply(lambda x: -1)

        return df

class DRT_LF(object):
    """
    Heuristic Based on Direct Retweets
    """
    def __init__(self,threshold,min_count):
        """
        """
        self.threshold=threshold
        self.min_count=min_count
    
    def __get_partisan_dist__(self,df):
        """
        """
        dist = df["matched_partisans"].value_counts().to_dict()
        norm_dist = {d: dist[d]/df.shape[0] for d in dist}
        norm_dist = defaultdict(int,norm_dist)

        conservative_score = norm_dist[float(2)] + norm_dist[float(3)]
        liberal_score = norm_dist[float(-2)] + norm_dist[float(-3)]

        return {"cons":round(conservative_score,4), "libs":round(liberal_score,4)}
    
    def label(self,df):
        """
        """
        
        if df.shape[0]>0:
        
            drt_df = df.loc[df["direct_retweet"]==True]

            if drt_df.shape[0]>self.min_count:
                ps_dist = self.__get_partisan_dist__(drt_df)

                if ps_dist["cons"]>ps_dist["libs"] and ps_dist["libs"]<=self.threshold:
                    df["ps_dist_drt_label"] = df["matched_partisans"].apply(lambda x: 1 if x<0 else 0)

                elif ps_dist["libs"]>ps_dist["cons"] and ps_dist["cons"]<=self.threshold:
                    df["ps_dist_drt_label"] = df["matched_partisans"].apply(lambda x: 1 if x>0 else 0)
                else:
                    df["ps_dist_drt_label"] = df["matched_partisans"].apply(lambda x: -1)
            else:
                df["ps_dist_drt_label"] = df["matched_partisans"].apply(lambda x: -1)
        
        return df

class PolFol_LF(object):
    """
    Heuristic Based on Followed Politicians
    """
    def __init__(self,pol_df,threshold,min_count,following_path="../../../Data/following_accounts/clean/"):
        """
        """
        self.threshold=threshold
        self.min_count=min_count
        self.pol_df = pol_df
        self.pol_df = self.pol_df.loc[self.pol_df["party_name"].isin(["Republican","Democrat"])].reset_index(drop=True)
        self.following_path=following_path
    
    def __get_dist__(self,overlapp):
        """
        """
        
        parties = []
        
        for acc in overlapp:
            matched = self.pol_df.loc[self.pol_df["user_name"]==acc,"party_name"].tolist()
            matched=matched[0]
            parties.append(matched)

        pcounts = Counter(parties)

        if "Republican" not in pcounts:
            pcounts["Republican"]=0

        if "Democrat" not in pcounts:
            pcounts["Democrat"]=0

        total = sum([pcounts[p] for p in pcounts])
        
        p_dist = {"cons":0.0,"libs":0.0}
        
        if total > 0:

            p_dist = {"cons":round(pcounts["Republican"]/total,4), "libs":round(pcounts["Democrat"]/total,4)}
        
        
        assert (p_dist["cons"] + p_dist["libs"] == 1) or (p_dist["cons"] + p_dist["libs"] == 0)
        
        return p_dist
    
    def label(self,df):
        """
        """
        if df.shape[0]>0:
        
            username = df["user"].iloc[0]
            if os.path.isfile(self.following_path+username+".pkl"):
                df_following = pd.read_pickle(self.following_path+username+".pkl")
                following_list = df_following["username"].tolist()

                pols_acc = self.pol_df["user_name"].tolist()

                overlapp = list(set(following_list).intersection(set(pols_acc)))

                if len(overlapp)>self.min_count:

                    ps_dist = self.__get_dist__(overlapp)

                    if ps_dist["cons"]>ps_dist["libs"] and ps_dist["libs"]<=self.threshold:
                        df["pol_fol_label"] = df["matched_partisans"].apply(lambda x: 1 if x<0 else 0)
                    elif ps_dist["libs"]>ps_dist["cons"] and ps_dist["cons"]<=self.threshold:
                        df["pol_fol_label"] = df["matched_partisans"].apply(lambda x: 1 if x>0 else 0)
                    else:
                        df["pol_fol_label"] = df["matched_partisans"].apply(lambda x: -1)

                else:
                    df["pol_fol_label"] = df["matched_partisans"].apply(lambda x: -1)

                return df

            else:
                df["pol_fol_label"] = df["matched_partisans"].apply(lambda x: -1)

                return df
        else:
            return df

class NSFol_LF(object):
    """
    Heuristic based on Followed News Sources
    """
    def __init__(self,news_df,threshold,min_count,following_path="../../../Data/following_accounts/clean/"):
        """
        """
        self.threshold=threshold
        self.min_count=min_count
        self.news_df = news_df
        self.news_df_twh = news_df.explode("Twitter Handle")
        self.following_path=following_path
        
    
    def __get_dist__(self,overlapp):
        """
        """
        
        news_partisan = []
        for acc in overlapp:
            matched = self.news_df_twh.loc[self.news_df_twh["Twitter Handle"]==acc,"Partisan Score"].tolist()
            matched = matched[0]
            if matched != 0:
                news_partisan.append(matched)

        pcounts = Counter(news_partisan)

        total_counts = sum([p[1] for p in pcounts.items()])

        cons_score = 0
        libs_score = 0

        for s in [-3,-2,-1,1,2,3]:
            if s in pcounts:
                if s>0:
                    cons_score+= pcounts[s]
                else:
                    libs_score+= pcounts[s]

        if total_counts > 0:
            cons_score = cons_score/total_counts
            libs_score = libs_score/total_counts
        
        assert (cons_score + libs_score == 1) or (cons_score+libs_score==0)
        
        ns_dist = {"cons":round(cons_score,4), "libs":round(libs_score,4)}
        
        return ns_dist
    
    def label(self,df):
        """
        """
        if df.shape[0]>0:
        
            username = df["user"].iloc[0]
            if os.path.isfile(self.following_path+username+".pkl"):
                df_following = pd.read_pickle(self.following_path+username+".pkl")
                following_list = df_following["username"].tolist()

                news_accs = self.news_df_twh["Twitter Handle"].tolist()

                overlapp = list(set(following_list).intersection(set(news_accs)))

                if len(overlapp)>self.min_count:

                    ps_dist = self.__get_dist__(overlapp)

                    if ps_dist["cons"]>ps_dist["libs"] and ps_dist["libs"]<=self.threshold:
                        df["ns_fol_label"] = df["matched_partisans"].apply(lambda x: 1 if x<0 else 0)
                    elif ps_dist["libs"]>ps_dist["cons"] and ps_dist["cons"]<=self.threshold:
                        df["ns_fol_label"] = df["matched_partisans"].apply(lambda x: 1 if x>0 else 0)
                    else:
                        df["ns_fol_label"] = df["matched_partisans"].apply(lambda x: -1)

                else:
                    df["ns_fol_label"] = df["matched_partisans"].apply(lambda x: -1)

                return df

            else:
                df["ns_fol_label"] = df["matched_partisans"].apply(lambda x: -1)

                return df
        
        else:
            return df
        



class TweetLevelLF(object):
    """
    Labelling Function based on Text Features
    """
    
    def __init__(self):
        """
        """
        
        self.direct_matches = {"mocking":["cover up",
                                        "covering up",
                                        "shameful reporting",
                                        "fakenews",
                                        "fraud news",
                                        "racist news",
                                        "fraud network",
                                        "racist network",
                                        "not reporting",
                                        "untrusted news",
                                        "shit news",
                                        "half truths",
                                        "tell the truth",
                                        "cant handle the truth",
                                        "bunch of crap",
                                        "stop lying",
                                        "brainwashed",
                                        "misinformation",
                                        "disinformation",
                                        "exaggerations",
                                        "scaremongering",
                                        "propaganda",
                                        "fearmongering",
                                        "hypocrisy",
                                        "boycott"],
                               "non-mocking":["watch this",
                                              "must watch",
                                              "live update",
                                              "listen to",
                                              "please read",
                                              "read this",
                                              "must read",
                                              "worth reading",
                                              "please share",
                                              "study finds",
                                              "top stories",
                                              "top story",
                                              "shocking news"]}
        
        self.conditional_matches = {"mocking":{"false":["news","stories","reporting","narrative","media"], 
                                               "fake":["news","reports","stories","story","report","media","network"],
                                               "hoax":["news","reports","stories","story","report","media"],
                                               "conspiracy":["theories","theory"],
                                               "fictitious":["news","report","story","narrative","media"],
                                               "misrepresent":["news","facts","truth","story","report","narrative","media"], 
                                               "misinform":["public,people,america"],
                                               "exaggerate":["news","report","story","narrative"],
                                               "mislead":["public,people,america"],
                                               "made up":["lies","crap","shit"],
                                               "make up":["lies","crap","shit"],
                                               "brainwash":["people","public","america"],
                                               "spread":["lies","propaganda","conspiracies","shit","fear"],
                                               "deceive":["people","public","america"],
                                               "biased" :["news","report","narrative","network","media","shit"],
                                               "one sided":["news","report","narrative","network","media"],
                                               "bullshit":["news","report","narrative","network","media"],
                                               "crap":["news","report","narrative","network","media"],
                                               "shit":["news","report","narrative","network","media"],
                                               "garbage":["news","report","narrative","network","media"]},
                                    "non-mocking":{"breaking":["news","exclusive","report","story"],
                                                  "watch":["now","live"],
                                                  "good":["news","report","story","journalism","narrative","article","piece","video"],
                                                  "great":["news","report","story","journalism","narrative","article","piece","video"],
                                                  "best":["news","report","video"],
                                                  "inspiring":["news","report","story","journalism","narrative","article","piece","video"],
                                                  "incredible":["news","report","story","journalism","narrative","article","piece","video"],
                                                  "real":["news","report","story","journalism","narrative","article","piece"],
                                                   "thanks":["news","report","story","journalism","narrative","article","piece"],
                                                  "thx":["news","report","story","journalism","narrative","article","piece"]},
                                                  "latest":["news","report","story","narrative","article","piece","scoop"],
                                                  "fantastic":["news","report","story","narrative","article","piece","scoop"]}
        
        
        
        self.pattern_matches = {"mocking":["expose @NS",
                                         "exposing @NS", 
                                         "exposes @NS", 
                                         "@NS exposed",
                                         "@NS sucks",
                                         "@NS is a joke",
                                         "@NS fuck you",
                                         "fuck you @NS",
                                         "screw you @NS",
                                         "@NS screw you",
                                         "fuck @NS",
                                         "@NS crap",
                                         "@NS is crap",
                                         "crap from @NS",
                                         "@NS should fire",
                                         "cant trust @NS",
                                         "can not trust @NS",
                                         "dont trust @NS",
                                         "do not trust @NS"],
                                
                                "non-mocking":["via @NS"]}
        
        self.punct_table = translation_init(exceptions=['?','!', '@', '#'])
        
    
    def check_direct_matches(self,tweet_row,check_type="mocking"):
        """
        """
        rule_fired = check_type
        if len(tweet_row.matched_mentions)<=0:
            return False
        
        candidates = self.direct_matches[check_type]
        
        text = tweet_row.text
        
        if tweet_row.tweet_type == "retweeted" and len(tweet_row.referenced_text)>0:
            
            text = tweet_row.referenced_text[0]
        
        text = remove_punctuations(text,self.punct_table)
        
        # check for matches
        for c in candidates:
            if c in text or c in text.lower():
                return True
        
        return False
            
    
    def check_conditional_matches(self,tweet_row,check_type="mocking"):
        """
        """
        rule_fired = check_type
        if len(tweet_row.matched_mentions)<=0:
            return False
            
        candidates = self.conditional_matches[check_type]
        
        text = tweet_row.text
        
        if tweet_row.tweet_type == "retweeted" and len(tweet_row.referenced_text)>0:
            
            text = tweet_row.referenced_text[0]
        
        text = remove_punctuations(text,self.punct_table)
        
        # check for matches
        # remove mentions to prevent miss matches
        text = re.sub(r'@\w+', '',text)
        for c in candidates:        
            if c in text or c in text.lower():
                for cw in candidates[c]:
                    if (cw in text or cw in text.lower()) and (c in text or c in text.lower()):
                        return True
        
        return False
                
    
    def check_pattern_matches(self,tweet_row,check_type="mocking"):
        """
        """
        rule_fired = check_type
        if len(tweet_row.matched_mentions)<=0:
            return False
        
        candidates = self.pattern_matches[check_type]
        
        text = tweet_row.text
        
        if tweet_row.tweet_type == "retweeted" and len(tweet_row.referenced_text)>0:
            
            text = tweet_row.referenced_text[0]
        
        extracted_ns_mentions = tweet_row.matched_mentions
        
        text = remove_punctuations(text,self.punct_table)
        
        # check for matches
        for p in candidates:
            for e in extracted_ns_mentions:
                pat = p.replace("NS",e)
                if pat in text or pat.lower() in text.lower():
                    return True
                
        return False
    
    def decide_mocking(self,tweet_row):
        """
        """
        verdicts = [tweet_row.text_lf_dm_mocking,
                    tweet_row.text_lf_cm_mocking,
                    tweet_row.text_lf_pm_mocking]
        
        return any(verdicts)
    
    def decide_non_mocking(self,tweet_row):
        """
        """
        verdicts = [tweet_row.text_lf_dm_non_mocking,
                    tweet_row.text_lf_cm_non_mocking,
                    tweet_row.text_lf_pm_non_mocking]
        
        return any(verdicts)
    
    def decide_label(self,tweet_row):
        """
        """
        mocking_label = tweet_row.tlf_mocking_label
        non_mocking_label = tweet_row.tlf_non_mocking_label
        
        # if tweet fulfills both conditions, default to mocking
        if mocking_label and non_mocking_label:
            return 1
        elif mocking_label and not non_mocking_label:
            return 1
        elif not mocking_label and non_mocking_label:
            return 0
        else:
            # when both are false (nothing matched)
            return -1
            
    
    def label(self,df):
        """
        """
        if df.shape[0]>0:
        
            # labelling the mocking class
            # for this only consider tweets with a news mention and only consider main tweet text
            df["text_lf_dm_mocking"] = df.apply(lambda x: self.check_direct_matches(x,check_type="mocking"),axis=1)
            df["text_lf_cm_mocking"] = df.apply(lambda x: self.check_conditional_matches(x,check_type="mocking"),axis=1)
            df["text_lf_pm_mocking"] = df.apply(lambda x : self.check_pattern_matches(x,check_type="mocking"),axis=1)
            df["tlf_mocking_label"] = df.apply(lambda x: self.decide_mocking(x),axis=1)


            # labelling non-mocking class
            # consider all tweets (mention or url of news source) and only consider main tweet text
            df["text_lf_dm_non_mocking"] = df.apply(lambda x: self.check_direct_matches(x,check_type="non-mocking"),axis=1)
            df["text_lf_cm_non_mocking"] = df.apply(lambda x: self.check_conditional_matches(x,check_type="non-mocking"),axis=1)
            df["text_lf_pm_non_mocking"] = df.apply(lambda x : self.check_pattern_matches(x,check_type="non-mocking"),axis=1)
            df["tlf_non_mocking_label"] = df.apply(lambda x: self.decide_non_mocking(x),axis=1)

            # final label
            df["keyword_label"] = df.apply(lambda x: self.decide_label(x),axis=1)
        
        return df