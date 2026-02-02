from inference_utils import filter_user_df
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
from collections import defaultdict
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import itertools
from sklearn.metrics import roc_curve

def get_binned_dfs(df,bin_ranges):
    """
    """
    binned_dfs = []
    for r in bin_ranges:
        min_ = r[0]
        max_ = r[1]
        
        bin_df = df.loc[(df["avg_stance"]>=min_) & (df["avg_stance"]<max_)]
        
        binned_dfs.append(bin_df)
    
    return binned_dfs

def get_user_distrust(user_df,min_size=30):
    """
    """
    
    user_df = filter_user_df(user_df)
    
    if user_df.shape[0]>min_size:
        
        user_df_distrusted = user_df.loc[user_df["model_preds"]==1]
        user_df_distrusted_lib = user_df_distrusted.loc[user_df_distrusted["matched_partisans"]<0]
        user_df_distrusted_cons = user_df_distrusted.loc[user_df_distrusted["matched_partisans"]>0]
        user_df_lib_all = user_df.loc[user_df["matched_partisans"]<0]
        user_df_cons_all = user_df.loc[user_df["matched_partisans"]>0]
        user_df_drt = user_df.loc[user_df["direct_retweet"]==True]
        
        distrust_cons_ratio = 0.0
        distrust_libs_ratio = 0.0
        
        if user_df_cons_all.shape[0]>0:
            distrust_cons_ratio = user_df_distrusted_cons.shape[0]/user_df_cons_all.shape[0]
        if user_df_lib_all.shape[0]>0:
            distrust_libs_ratio = user_df_distrusted_lib.shape[0]/user_df_lib_all.shape[0]
        
        if user_df_drt.shape[0]>0:
            
        
            return {"d-cons":distrust_cons_ratio,
                    "d-lib":distrust_libs_ratio,
                    "avg_stance":user_df_drt["matched_partisans"].mean(),
                    "user-seq":user_df.shape[0]}
        
        return None
    
    else:
        return None
        

def get_distrust_scores(all_data,min_size=30):
    """
    """
    all_scores = []
    skipped_users = 0
    seq_sizes = []
    for usr,user_df in tqdm.tqdm(all_data.groupby("user")):
        
        user_scores = get_user_distrust(user_df,min_size=min_size)
        
        if user_scores != None:
            seq_sizes.append(user_scores["user-seq"])
            score_df = pd.DataFrame([user_scores])
            all_scores.append(score_df)
        
        else:
            skipped_users+=1
    
    all_scores=pd.concat(all_scores)
    
    print(f"Number of Users Skipped due to small seq length : {skipped_users}")
    
    return all_scores

def plot_distrust_ratios_custom(all_labelled_data,bin_ranges=[(-3,-2),(-2,-1),(-1,1),(1,2),(2,3)],error="sem"):
    """
    """
    avg_stance = []
    lib_d_ratio = []
    cons_d_ratio = []
    
    lib_d_errors = []
    lib_d_std_devs = []
    
    cons_d_errors = []
    cons_d_std_devs = []
    
    distrust_scores = get_distrust_scores(all_labelled_data,min_size=30)
    binned_dfs = get_binned_dfs(distrust_scores,bin_ranges)
    
    for df_q in binned_dfs:
        
        avg_stance.append(df_q["avg_stance"].mean())
        lib_d_ratio.append(df_q["d-lib"].mean())
        cons_d_ratio.append(df_q["d-cons"].mean())
        lib_d_errors.append(df_q["d-lib"].sem())
        lib_d_std_devs.append(df_q["d-lib"].std())
        cons_d_errors.append(df_q["d-cons"].sem())
        cons_d_std_devs.append(df_q["d-cons"].std())
    
    
    fig,ax_ = plt.subplots(1,2,figsize=(7,3),gridspec_kw={'width_ratios': [4, 1]})
    
    axes = ax_.ravel()
    
    ax = axes[0]
    ax2 = axes[1]
    
    width=0.35
    x = np.arange(len(avg_stance))
    
    yerror_lib = None
    yerror_cons = None
    
    if error == "sem":
        yerror_lib = lib_d_errors
        yerror_cons = cons_d_errors
        
    if error == "std":
        yerror_lib = lib_d_std_devs
        yerror_cons = cons_d_std_devs
    
    rects1 = ax.bar(x - width/2, lib_d_ratio,width,yerr=yerror_lib,color="maroon", alpha=0.8,ecolor='black', capsize=5, edgecolor="black",label='Criticism Towards Liberal Media')
    rects2 = ax.bar(x + width/2, cons_d_ratio,width,yerr=yerror_cons,color="teal", alpha=0.8,ecolor='black', capsize=5, edgecolor="black",label='Criticism Towards Conservative Media')
    
    ax.set_ylabel('Criticism Ratio',fontsize=14,labelpad=10)
    ax.set_xlabel("Average Partisan Stance\n"+r"$\leftarrow ---$"+" Liberal   Conservative"+r"$--- \rightarrow$",fontsize=14,labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bin_ranges])
    
    for axis in ['top', 'bottom', 'left', 'right']:

        ax.spines[axis].set_linewidth(1.5) 
    
    ax.grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    ax.grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)
    ax.set_ylim([0.0,0.4])
    
    
    ax.legend(fontsize=10,loc="upper center")
    
    overall_lib_d_ratio = distrust_scores["d-lib"].mean()
    overall_cons_d_ratio = distrust_scores["d-cons"].mean()
    overall_lib_d_ratio_error = distrust_scores["d-lib"].sem()
    overall_cons_d_ratio_error = distrust_scores["d-cons"].sem()
    
    ax2.bar(1,overall_lib_d_ratio,width,yerr=overall_lib_d_ratio_error,color="maroon", alpha=0.8,ecolor='black', capsize=5, edgecolor="black")
    ax2.bar(0,overall_cons_d_ratio,width,yerr=overall_cons_d_ratio_error,color="teal", alpha=0.8,ecolor='black', capsize=5, edgecolor="black")
    
    ax2.grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    ax2.grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_xticks([])
    for axis in ['top', 'bottom', 'left', 'right']:

        ax2.spines[axis].set_linewidth(1.5) 
    
    
    fig.savefig("latest_graphs/distrust_ratio.png",format="png",dpi=400,bbox_inches = "tight")
    
    fig.savefig("latest_graphs/distrust_ratio.pdf",format="pdf",dpi=400,bbox_inches = "tight")
    
    plt.show()


def get_news_source_mocking_dist(labelled_df):
    """
    """
    count_dist = defaultdict(lambda : {"trust":0.0, "distrust":0.0})
    
    labelled_df = filter_user_df(labelled_df)
    
    labelled_df = labelled_df.explode("matched_sources").reset_index(drop=True)
    
    for ind,row in tqdm.tqdm(labelled_df.iterrows()):
        s = row.matched_sources
        m = row.model_preds
        if m == 1:
            count_dist[s]["distrust"]+=1.0
        else:
            count_dist[s]["trust"]+=1.0
    
    return count_dist


def single_plot_pop(ns_pop,ax,title,color,clean_sources):
    """
    """
    x = []
    y = []
    
    
    for ns in ns_pop:
        x.append(ns[0].replace("The",""))
        y.append(ns[1]["distrust"]/(ns[1]["distrust"]+ns[1]["trust"]))
    
    x, y = zip(*sorted(zip(x, y),key=lambda x: x[1],reverse=True))
    
    x = [clean_sources[i] if i in clean_sources else i for i in x] 
    
    x_ =[]
    y_ = []
    for a,b in zip(x,y):
        if b>0.0:
            x_.append(a)
            y_.append(b)
    
    ax.bar(x_,y_,color=color,alpha=0.9,edgecolor="black")
    ax.set_xticklabels(x, rotation=90,fontsize=11)
    ax.set_title(title,fontsize=13)
    
    for axis in ['top', 'bottom', 'left', 'right']:

        ax.spines[axis].set_linewidth(1.5) 
    
    ax.grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    ax.grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)
    ax.tick_params(axis='x', which='minor',length=0)



clean_sources = {"CNN Online News": "CNN", 
                 "dailynewsbin":"Daily News Bin",
                 "firstpost":"First Post",
                 "palmerreport":"Palmer Report",
                 "occupydemocrats": "Occupy Democrats",
                 "newspunch":"News Punch",
                 "crooksandliars":"Crooks and Liars",
                 "bipartisanreport":"Bipartisan Report",
                 "indiatimes": "IndiaTimes",
                 "collective-evolution": "Collective Evolution",
                 "usuncut":"US Uncut",
                 "Fox News Online News": "Fox News",
                 "One America News Network OAN":"OANN",
                 "Blazecom":"Blaze.com",
                 "chicksontheright":"Chicks On the Right",
                 "disclose":"Disclose",
                 "allenbwest":"Allen B West",
                 "theconservativetreehouse":"Conservative Tree House",
                 "wearechange":"We are Change",
                 "express":"Express",
                 "barenakedislam":"Bare Naked Islam",
                 "conservativetribune":"Conservative Tribune",
                 "beforeitsnews": "Before It's News",
                 "clashdaily":"Clash Daily","HuffPost": "Huffington Post"}


def plot_by_popularity(ns_count_dist,news_df,clean_sources,top_n=30):
    """
    """
    # sort by popularity
    sorted_ns_counts = dict(sorted(ns_count_dist.items(),key=lambda item: item[1]["trust"]+item[1]["distrust"],reverse=True))
    
    sources_2_partisan = {x:y for x,y in zip(news_df["Source"].tolist(),news_df["Partisan Score"].tolist())}
    
    top_lib_sources = [(s,sorted_ns_counts[s]) for s in sorted_ns_counts if sources_2_partisan[s]==-2][:top_n]
    top_cons_sources = [(s,sorted_ns_counts[s]) for s in sorted_ns_counts if sources_2_partisan[s]==2][:top_n]
    
    print(top_cons_sources)
    top_lib_fake_sources = [(s,sorted_ns_counts[s]) for s in sorted_ns_counts if sources_2_partisan[s]==-3][:top_n]
    top_cons_fake_sources = [(s,sorted_ns_counts[s]) for s in sorted_ns_counts if sources_2_partisan[s]==3][:top_n]
    
    fig,ax = plt.subplots(1,4,figsize=(12,2),sharey=True)
    axes = ax.ravel()
    single_plot_pop(top_lib_sources,axes[1],"Liberal (-2)","tab:blue",clean_sources)
    single_plot_pop(top_lib_fake_sources,axes[0],"Unreliable Liberal (-3)","tab:cyan",clean_sources)
    single_plot_pop(top_cons_sources,axes[2],"Conservative (+2)","tab:red",clean_sources)
    single_plot_pop(top_cons_fake_sources,axes[3],"Unreliable Conservative (+3)","tab:orange",clean_sources)
    
    axes[0].set_ylabel("Criticism Ratio",fontsize=13)
    
    axes[1].yaxis.set_ticks_position('none')
    axes[2].yaxis.set_ticks_position('none')
    axes[3].yaxis.set_ticks_position('none')
    
    fig.subplots_adjust(wspace=0.03,hspace=0.0)
    
    fig.savefig("latest_graphs/news_mocking_amount.png",format="png",dpi=400,bbox_inches = "tight")
    
    fig.savefig("latest_graphs/news_mocking_amount.pdf",format="pdf",dpi=400,bbox_inches = "tight")
    
    plt.show()



def calc_nentropy(stances,m=4):
    """
    """
    if m == 4:
        stance_dict = {-2:0,-3:0,2:0,3:0}

        for s in stances:
            stance_dict[s]+=1

        total = sum(stance_dict.values())

        num = 0
        for s in stance_dict:
            if stance_dict[s]>0:
                num += (stance_dict[s]/total)*np.log2(stance_dict[s]/total)
        
        if num == 0:
            return 0
        return -1 * num/np.log2(4)
    
    if m == 2:
        stance_dict = {0:0,1:0} # liberal -> 0, conservative -> 1
        
        for s in stances:
            if s>0:
                stance_dict[1]+=1
            else:
                stance_dict[0]+=1
        
        total = sum(stance_dict.values())

        num = 0
        for s in stance_dict:
            if stance_dict[s]>0:
                num += (stance_dict[s]/total)*np.log2(stance_dict[s]/total)
        
        if num == 0:
            return 0
        return -1 * num/np.log2(2)


def get_user_entropies_before_after(all_labelled_data):
    """
    """
    user_entropies_before = []
    user_entropies_after = []

    for usr,df_user in tqdm.tqdm(all_labelled_data.groupby("user")):
        
        df_user = filter_user_df(df_user)
        
        if df_user.shape[0]>30:

            stances_before = df_user["matched_partisans"].tolist()

            stances_wo_distrust = df_user.loc[df_user["model_preds"]==0,"matched_partisans"].tolist()

            entropy_before = calc_nentropy(stances_before,m=4)

            entropy_after = calc_nentropy(stances_wo_distrust,m=4)

            user_entropies_before.append(entropy_before)

            user_entropies_after.append(entropy_after)
    
    return user_entropies_before, user_entropies_after


def plot_stance_entropy(all_labelled_data):
    """
    """
    fig,ax = plt.subplots(1,1,figsize=(4,4))

    mly_major = MultipleLocator(0.2)
    mly_minor = MultipleLocator(0.1)
    mlx_major = MultipleLocator(0.2)
    mlx_minor = MultipleLocator(0.1)
    
    user_entropies_before, user_entropies_after = get_user_entropies_before_after(all_labelled_data)

    sns.kdeplot(np.array(user_entropies_before),bw_adjust=.5,ax=ax,color="orange",fill=True,alpha=0.3,label="Before(mean=0.358)")
    sns.kdeplot(np.array(user_entropies_after),bw_adjust=.5,ax=ax,color="blue",fill=True,alpha=0.3,label="After(mean=0.339)")
    
    ax.axvline(np.mean(np.array(user_entropies_before)),linestyle="--",color="orange",alpha=0.95)
    ax.axvline(np.mean(np.array(user_entropies_after)),linestyle="--",color="blue",alpha=0.95)
    
    print(np.mean(np.array(user_entropies_before)))
    print(np.mean(np.array(user_entropies_after)))

    ax.set_ylabel("Density",fontsize=14,labelpad=10)

    ax.set_xlabel("Normalized Stance Entropy (NSE)",fontsize=14,labelpad=10)

    ax.grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    ax.grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)

    ax.xaxis.set_minor_locator(mlx_minor)
    ax.xaxis.set_major_locator(mlx_major)
    ax.yaxis.set_major_locator(mly_major)
    ax.yaxis.set_minor_locator(mly_minor)


    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.legend(loc="upper right",ncol=1,fontsize=9)
    
    for axis in ['top', 'bottom', 'left', 'right']:

        ax.spines[axis].set_linewidth(1.5) 
    
    fig.savefig('latest_graphs/stance_entropy.png', format='png', dpi=400,bbox_inches="tight")
    fig.savefig('latest_graphs/stance_entropy.pdf', format='pdf', dpi=400,bbox_inches="tight")
    plt.show()
    
    
def split_data_by_avg_stance(all_data,extreme=False):
    """
    """
    liberal_users = []
    conservative_users = []
    for usr,user_df in tqdm.tqdm(all_data.groupby("user")):
        
        f_udf = filter_user_df(user_df)
        
        if f_udf.shape[0]>0:
            
            avg_stance = f_udf.loc[f_udf["direct_retweet"]==True,"matched_partisans"].mean()
            
            if not extreme:
                if avg_stance > 0:
                    conservative_users.append(f_udf)

                if avg_stance < 0:
                    liberal_users.append(f_udf)
            
            if extreme:
                if avg_stance > +1:
                    conservative_users.append(f_udf)

                if avg_stance < -1:
                    liberal_users.append(f_udf)
    
    print(f"Number of Liberal Users based on Avg DRT Stance : {len(liberal_users)}")
    print(f"Number of Conservative Users based on Avg DRT Stance : {len(conservative_users)}")
    
    conservative_df = pd.concat(conservative_users).reset_index(drop=True)
    liberal_df = pd.concat(liberal_users).reset_index(drop=True)
    
    print(f"Size of Liberal User Observations : {len(liberal_df)}")
    print(f"Size of Conservative User Observations : {len(conservative_df)}")
            
    return liberal_df, conservative_df



def get_resampled_distrust_percentage_user_alt(data,resample_window="1M",min_year=2012):
    """
    """
    data = filter_user_df(data)
    
    distrust_only = data.loc[data["model_preds"]==1]
    distrust_only_filtered = distrust_only.reset_index(drop=True)
    distrust_only_filtered["created_at"] = pd.to_datetime(distrust_only_filtered["created_at"])
    distrust_only_filtered = distrust_only_filtered.sort_values(by="created_at").reset_index(drop=True)
    distrust_only_filtered_resampled = distrust_only_filtered.set_index("created_at")
    distrust_only_filtered_resampled = distrust_only_filtered_resampled.loc[distrust_only_filtered_resampled.index >= f"{min_year}-01-01"]
    distrust_only_filtered_resampled = distrust_only_filtered_resampled.resample(f"{resample_window}").count()
    
    data_ts = data.copy(deep=True)
    data_ts["created_at"] = pd.to_datetime(data_ts["created_at"])
    data_ts = data_ts.sort_values(by="created_at").reset_index(drop=True)
    data_ts = data_ts.set_index("created_at")
    data_ts = data_ts.loc[data_ts.index > f"{min_year}-01-01"]
    data_ts =data_ts.resample(f"{resample_window}").count()
    
    print(f"Distrust shape : {distrust_only_filtered_resampled.shape}")
    print(f"All shape : {data_ts.shape}")
    
    ratios_df = distrust_only_filtered_resampled.div(data_ts)
    
    return ratios_df



def get_resampled_distrust_percentage_engagements(data,resample_window="1M",min_year=2012):
    """
    """
    data = filter_user_df(data)
    
    left_distrusted = data.loc[(data["model_preds"]==1) & (data["matched_partisans"]<0)].reset_index(drop=True)
    right_distrusted = data.loc[(data["model_preds"]==1) & (data["matched_partisans"]>0)].reset_index(drop=True)
    
    left_distrusted["created_at"] = pd.to_datetime(left_distrusted["created_at"])
    left_distrusted = left_distrusted.sort_values(by="created_at").reset_index(drop=True)
    left_distrusted = left_distrusted.set_index("created_at")
    left_distrusted = left_distrusted.resample(f"{resample_window}",label="left", closed="left").count()
    left_distrusted = left_distrusted.loc[left_distrusted.index >= f"{min_year}-01-01"]
    
    right_distrusted["created_at"] = pd.to_datetime(right_distrusted["created_at"])
    right_distrusted = right_distrusted.sort_values(by="created_at").reset_index(drop=True)
    right_distrusted = right_distrusted.set_index("created_at")
    right_distrusted = right_distrusted.resample(f"{resample_window}",label="left", closed="left").count()
    right_distrusted = right_distrusted.loc[right_distrusted.index >= f"{min_year}-01-01"]
    
    all_distrust = data.loc[data["model_preds"]==1].reset_index(drop=True)
    all_distrust["created_at"] = pd.to_datetime(all_distrust["created_at"])
    all_distrust = all_distrust.sort_values(by="created_at").reset_index(drop=True)
    all_distrust = all_distrust.set_index("created_at")
    all_distrust = all_distrust.resample(f"{resample_window}",label="left", closed="left").count()
    all_distrust = all_distrust.loc[all_distrust.index >= f"{min_year}-01-01"]
    
    data_ts = data.copy(deep=True)
    data_ts_liberal = data_ts.loc[data_ts["matched_partisans"]<0].reset_index(drop=True)
    data_ts_conservative = data_ts.loc[data_ts["matched_partisans"]>0].reset_index(drop=True)
    
    data_ts_liberal["created_at"] = pd.to_datetime(data_ts_liberal["created_at"])
    data_ts_liberal = data_ts_liberal.sort_values(by="created_at").reset_index(drop=True)
    data_ts_liberal = data_ts_liberal.set_index("created_at")
    data_ts_liberal =data_ts_liberal.resample(f"{resample_window}",label="left", closed="left").count()
    data_ts_liberal = data_ts_liberal.loc[data_ts_liberal.index > f"{min_year}-01-01"]
    
    data_ts_conservative["created_at"] = pd.to_datetime(data_ts_conservative["created_at"])
    data_ts_conservative = data_ts_conservative.sort_values(by="created_at").reset_index(drop=True)
    data_ts_conservative = data_ts_conservative.set_index("created_at")
    data_ts_conservative =data_ts_conservative.resample(f"{resample_window}",label="left", closed="left").count()
    data_ts_conservative = data_ts_conservative.loc[data_ts_conservative.index > f"{min_year}-01-01"]
    
    data_ts["created_at"] = pd.to_datetime(data_ts["created_at"])
    data_ts = data_ts.sort_values(by="created_at").reset_index(drop=True)
    data_ts = data_ts.set_index("created_at")
    data_ts =data_ts.resample(f"{resample_window}",label="left", closed="left").count()
    data_ts = data_ts.loc[data_ts.index > f"{min_year}-01-01"]
    
    left_distrust_rat = left_distrusted/data_ts_liberal
    right_distrust_rat = right_distrusted/data_ts_conservative
    overall_distrust_rat = all_distrust/data_ts
    
    
    return left_distrust_rat, right_distrust_rat, overall_distrust_rat


def plot_ts_distrust_user(ax,ratios_liberal,ratios_cons):
    """
    """
    mly_major = MultipleLocator(0.02)
    mly_minor = MultipleLocator(0.01)
    
    ax.plot(ratios_liberal.index.values,
            ratios_liberal["model_preds"].tolist(),
            marker=".",
            markeredgecolor="black",
            color="tab:blue",
            label="liberal-users",
            linestyle="-")
    
    ax.plot(ratios_cons.index.values,
            ratios_cons["model_preds"].tolist(),
            marker=".",
            markeredgecolor="black",
            color="tab:red",
            label="conservative-users",linestyle="-")
    
    ax.fill_between(ratios_liberal.index.values,  ratios_liberal["model_preds"].tolist(), ratios_cons["model_preds"].tolist(), color='#00afb9', alpha=0.3)

    ax.yaxis.set_major_locator(mly_major)
    ax.yaxis.set_minor_locator(mly_minor)
    ax.grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    ax.grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)

    # ax.set_xlabel("Year",fontsize=13)
    ax.set_ylabel("Criticism Ratio",fontsize=13,labelpad=10)


def plot_ts_distrust_engagements(ax,left,right):
    """
    """
    mly_major = MultipleLocator(0.02)
    mly_minor = MultipleLocator(0.01)
    
    ax.plot(left.index.values,
            left["model_preds"].tolist(),
            marker=".",
            markeredgecolor="black",
            color="tab:blue",
            label="liberal-media",
            linestyle="-")
    
    ax.plot(right.index.values,
            right["model_preds"].tolist(),
            marker=".",
            markeredgecolor="black",
            color="tab:red",
            label="conservative-media",
            linestyle="-")
    
    
    ax.fill_between(left.index.values,  left["model_preds"].tolist(), right["model_preds"].tolist(), color='#219ebc', alpha=0.3)
    
    ax.set_ylabel("Criticism Ratio",fontsize=13,labelpad=10)
    
    ax.yaxis.set_major_locator(mly_major)
    ax.yaxis.set_minor_locator(mly_minor)
    ax.grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    ax.grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)

def criticism_by_time(all_labelled_data):
    """
    """
    fig,ax = plt.subplots(2,1,figsize=(6,6),sharex=True)

    axes = ax.ravel()
    
    liberal_df, conservative_df = split_data_by_avg_stance(all_labelled_data,extreme=False)
    
    ratios_df_liberal = get_resampled_distrust_percentage_user_alt(liberal_df,resample_window="1MS",min_year=2013)
    ratios_df_cons = get_resampled_distrust_percentage_user_alt(conservative_df,resample_window="1MS",min_year=2013)

    left_distrust_rat, right_distrust_rat,over_dis_rat = get_resampled_distrust_percentage_engagements(all_labelled_data,
                                                                                        resample_window="1MS",
                                                                                        min_year=2013)

    plot_ts_distrust_user(axes[0],ratios_df_liberal,ratios_df_cons)
    plot_ts_distrust_engagements(axes[1],left_distrust_rat,right_distrust_rat)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax[0].spines[axis].set_linewidth(1.5) 

    for axis in ['top', 'bottom', 'left', 'right']:
        ax[1].spines[axis].set_linewidth(1.5) 

    axes[0].legend(loc="upper left")
    axes[1].legend(loc="upper left")
    axes[0].xaxis.set_ticks_position('none')
    fig.subplots_adjust(wspace=0.0,hspace=0.15)
    axes[0].set_title("Source of Criticism")
    axes[1].set_title("Target of Criticism")
    axes[1].set_xlabel("Year",fontsize=13)

    fig.savefig('latest_graphs/distrust_by_time.png', format='png', dpi=400,bbox_inches="tight")
    fig.savefig('latest_graphs/distrust_by_time.pdf', format='pdf', dpi=400,bbox_inches="tight")
    plt.show()



def check_dates(dates):
    """
    """
    prev = dates[0]
    for d in dates[1:]:
        if d > prev:
            return False
        else:
            prev=d
    
    return True

def update_ps(trow):
    """
    """
    if trow.model_preds == 1:
        
        if trow.matched_partisans>=2:
            return "D&C"
        
        if trow.matched_partisans<=-2:
            return "D&L"
        
    else:
        return trow.matched_partisans

def get_seq_counts(df,states=[-2,"D&C",-3]):
    """
    """
    users_count = 0
    users_considered = 0
    users_total = 0
    
    for usr,user_df in df.groupby("user"):
        users_total+=1
        user_df = filter_user_df(user_df)
        user_df["ts_label"] = user_df.apply(lambda x: update_ps(x) ,axis=1)
        user_df["created_at"] = pd.to_datetime(user_df["created_at"])
        
        user_df = user_df.sort_values(by="created_at").reset_index(drop=True)
        dates = []
        for s in states:
            fo_d = user_df.loc[user_df["ts_label"]==s,"created_at"].min()
            if not pd.isnull(fo_d):
                dates.append((s,fo_d))
        
        if len(dates) == len(states):
            users_considered+=1
        
        dates_sorted = sorted(dates,key=lambda x: x[1],reverse=False)
        states_sorted = [d[0] for d in dates_sorted]
        
        if states_sorted == states:
            users_count+=1
            
    print(f"No of Users that follow given state sequence : {users_count}")
    print(f"Total available users : {users_total}")
    print(f"Total users who engage with given states : {users_considered}")


def get_first_occurence_patterns(all_labelled_data):
    """
    """
    all_possible_seq_liberal = [list(x) for x in itertools.permutations([-2,"D&C",-3])]
    all_possible_seq_conservative = [list(x) for x in itertools.permutations([2,"D&L",3])]
    
    liberal_df, conservative_df = split_data_by_avg_stance(all_labelled_data,extreme=False)
    
    print(f"\nLiberal User Group :\n")
    for s in all_possible_seq_liberal:
        print(s)
        get_seq_counts(liberal_df,states=s)
        print("\n")
    
    print(f"\nConservative User Group :\n")
    for s in all_possible_seq_conservative:
        print(s)
        get_seq_counts(conservative_df,states=s)
        print("\n")


def plot_roc_curves(pred_df,models=["text_model_user",
                                    "user_model_user",
                                    "text_model_text",
                                    "user_model_text",
                                    "text_model_union",
                                    "user_model_union",
                                    "combined_model_union",
                                    "combined_model_text",
                                    "combined_model_user"]):
    """
    """
    fig,ax = plt.subplots(1,3,figsize=(9,3),sharey=True)
    
    axes = ax.ravel()
    
    mly_major = MultipleLocator(0.2)
    mly_minor = MultipleLocator(0.1)
    mlx_major = MultipleLocator(0.2)
    mlx_minor = MultipleLocator(0.1)
    
    colors = {"text" :"tab:orange", "user":"tab:cyan", "union":"tab:pink"}
    for m in models:
        
        fpr, tpr, thresholds = roc_curve(pred_df["annotations"], pred_df[f"{m}_pred_prob"], pos_label=1)
        
        heuris_label = m.split("_")[-1]
        
        label = None
        if heuris_label == "text":
            label = '$\phi_{tt}$'
        if heuris_label == "user":
            label = '$\phi_{up}$'
        if heuris_label == "union":
            label = '$\phi_{un}$'
        
        if m.split("_")[0] == "text":
            
            axes[0].plot(fpr,tpr,color="black",alpha=0.9,linewidth=3.5)
            axes[0].plot(fpr,tpr,label=label,alpha=0.9,linewidth=3,color=colors[heuris_label])
        
        if m.split("_")[0] == "user":
            
            axes[1].plot(fpr,tpr,color="black",alpha=0.9,linewidth=3.5)
            axes[1].plot(fpr,tpr,label=label,alpha=0.9,linewidth=3,color=colors[heuris_label])
        
        if m.split("_")[0] == "combined":
            
            axes[2].plot(fpr,tpr,color="black",alpha=0.9,linewidth=3.5)
            axes[2].plot(fpr,tpr,label=label,alpha=0.9,linewidth=3,color=colors[heuris_label])
        
    
    axes[0].set_xlabel("FPR",fontsize=18)
    axes[1].set_xlabel("FPR",fontsize=18)
    axes[2].set_xlabel("FPR",fontsize=18)
    
    axes[0].set_ylabel("TPR",fontsize=18)
    
    
    axes[0].plot([0, 1], [0, 1], transform= axes[0].transAxes,linestyle="--",linewidth=2,color="black",alpha=0.7)
    axes[1].plot([0, 1], [0, 1], transform= axes[1].transAxes,linestyle="--",linewidth=2,color="black",alpha=0.7)
    axes[2].plot([0, 1], [0, 1], transform= axes[2].transAxes,linestyle="--",linewidth=2,color="black",alpha=0.7)
    
    axes[1].yaxis.set_ticks_position('none')
    axes[2].yaxis.set_ticks_position('none')
    
    axes[0].set_title("Text Network",fontsize=20)
    axes[1].set_title("User Network",fontsize=20)
    axes[2].set_title("Combined Network",fontsize=20)
    
    axes[0].xaxis.set_major_locator(mlx_major)
    axes[0].yaxis.set_major_locator(mly_major)
    axes[0].xaxis.set_minor_locator(mlx_minor)
    axes[0].yaxis.set_minor_locator(mly_minor)

    axes[0].grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    axes[0].grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)
    
    axes[1].xaxis.set_major_locator(mlx_major)
    axes[1].yaxis.set_major_locator(mly_major)
    axes[1].xaxis.set_minor_locator(mlx_minor)
    axes[1].yaxis.set_minor_locator(mly_minor)

    axes[1].grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    axes[1].grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)
    
    axes[2].xaxis.set_major_locator(mlx_major)
    axes[2].yaxis.set_major_locator(mly_major)
    axes[2].xaxis.set_minor_locator(mlx_minor)
    axes[2].yaxis.set_minor_locator(mly_minor)

    axes[2].grid(True,which="major", linewidth=0.8,linestyle='--',alpha=0.4)
    axes[2].grid(True,which="minor",linewidth=0.5,linestyle='--',alpha=0.4)
    
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    axes[2].set_aspect('equal')
    
    for axis in ['top', 'bottom', 'left', 'right']:

        ax[0].spines[axis].set_linewidth(1.5)
    
    for axis in ['top', 'bottom', 'left', 'right']:

        ax[1].spines[axis].set_linewidth(1.5)
    
    for axis in ['top', 'bottom', 'left', 'right']:

        ax[2].spines[axis].set_linewidth(1.5)
    
    ax[0].tick_params(axis='x', which='minor',length=0)
    ax[1].tick_params(axis='x', which='minor',length=0)
    ax[2].tick_params(axis='x', which='minor',length=0)
    
    ax[0].tick_params(axis='both', which='major',labelsize=16)
    ax[1].tick_params(axis='both', which='major',labelsize=16)
    ax[2].tick_params(axis='both', which='major',labelsize=16)
    
    fig.subplots_adjust(wspace=0.1,hspace=0.0)
    
    ax[2].legend(loc="lower right",fontsize=14)
    
    fig.savefig("latest_graphs/model_performance.png",format="png",dpi=400,bbox_inches = "tight")
    fig.savefig("latest_graphs/model_performance.pdf",format="pdf",dpi=400,bbox_inches = "tight")
    plt.show()