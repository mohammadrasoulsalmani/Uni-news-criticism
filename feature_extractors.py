from collections import defaultdict, Counter
import pandas as pd



class ExtractUserFeats(object):
    """
    Extract Features based on a User's Historic News Engagements
    """
    
    def __init__(self,
                 news_df,
                 pol_df,
                 following_path="../../../Data/following_accounts/clean/"):
        """
        """
        self.news_df = news_df
        self.news_df_twh = news_df.explode("Twitter Handle")
        self.pol_df = pol_df
        self.pol_df = self.pol_df.loc[self.pol_df["party_name"].isin(["Republican","Democrat"])].reset_index(drop=True)
        self.following_path = following_path
    
    def __get_partisan_dist__(self,df):
        """
        """
        dist = df["matched_partisans"].value_counts().to_dict()
        norm_dist = {d: dist[d]/df.shape[0] for d in dist}
        norm_dist = defaultdict(int,norm_dist)

        conservative_score = norm_dist[float(2)] + norm_dist[float(3)]
        liberal_score = norm_dist[float(-2)] + norm_dist[float(-3)]

        return {"cons":round(conservative_score,4), "lib":round(liberal_score,4)}

    
    def __get_direct_news_retweet_dist__(self,df):
        """
        """
        
        df_rtw = df.loc[df["direct_retweet"]==True]
        ps_dist = self.__get_partisan_dist__(df_rtw)
        return ps_dist
        
    
    def __get_accounts_followed__(self,following_list):
        """
        """
        pols_acc = self.pol_df["user_name"].tolist()
        
        news_accs = self.news_df_twh["Twitter Handle"].tolist()
        
        overlapp = list(set(following_list).intersection(set(pols_acc+news_accs)))
        
        return overlapp
        
    
    def __get_pol_followed__(self,following_list):
        """
        """
        pols_acc = self.pol_df["user_name"].tolist()
        
        overlapp = list(set(following_list).intersection(set(pols_acc)))
        
        verdict = False
        
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
        
        p_dist = {"cons":0.0,"lib":0.0}
        
        if total > 0:

            p_dist = {"cons":round(pcounts["Republican"]/total,4), "lib":round(pcounts["Democrat"]/total,4)}
        
        
        assert (p_dist["cons"] + p_dist["lib"] == 1) or (p_dist["cons"] + p_dist["lib"] == 0)

        
        return  p_dist
            
    
    def __get_ns_followed__(self,following_list):
        """
        """
        news_accs = self.news_df_twh["Twitter Handle"].tolist()
        
        overlapp = list(set(following_list).intersection(set(news_accs)))
        
        
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
        
        ns_dist = {"cons":round(cons_score,4), "lib":round(libs_score,4)}
            
            
        return ns_dist
        
    
    def get_news_source_frequencies(self,news_sources_matched):
        """
        news_sources_matched - list of list
        """
        # create a dict of ns->counts(how many times mentioned)
        sources = self.news_df["Source"].tolist()
        count_dict = {s:0 for s in sources}
        all_sources = [item for ns_l in news_sources_matched for item in ns_l]
        for a in all_sources:
            count_dict[a]+=1
        
        return count_dict

    
    def extract(self,df):
        """
        """
        username = df["user"].iloc[0]
        
        partisan_dist_drt = self.__get_direct_news_retweet_dist__(df)
        partisan_dist = self.__get_partisan_dist__(df)
   
        pol_dist = {"cons":0.0, "lib":0.0}
        ns_dist = {"cons":0.0, "lib":0.0}
        
        df_following = pd.read_pickle(self.following_path+username+".pkl")
        following_list = df_following["username"].tolist()
        pol_dist = self.__get_pol_followed__(following_list)
        ns_dist = self.__get_ns_followed__(following_list)
        
        accs_followed = self.__get_accounts_followed__(following_list)
            
        df["accounts_followed"] = df["matched_partisans"].apply(lambda x: accs_followed)

        df["partisan_dist"] = df["matched_partisans"].apply(lambda x: partisan_dist)

        df["drt_partisan_dist"] = df["matched_partisans"].apply(lambda x: partisan_dist_drt)

        df["pol_dist"] = df["matched_partisans"].apply(lambda x: pol_dist)

        df["ns_dist"] = df["matched_partisans"].apply(lambda x: ns_dist)
        
        freq_dict = self.get_news_source_frequencies(df["matched_sources"].tolist())
            
        df["news_source_frequency"] = [freq_dict]*df.shape[0]
        
        return df
    