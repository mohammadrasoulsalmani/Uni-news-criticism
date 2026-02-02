import pandas as pd

def get_coverage(data):
    """
    """
    label_columns =  ["ps_dist_label",
                      "ps_dist_drt_label",
                      "pol_fol_label",
                      "keyword_label",
                      "user_label",
                      "unanimous_label",
                      "union_heuristic"]
    
    total_size = data.shape[0]
    
    coverage_list = []
    
    for l in label_columns:
        coverage_dict = {}
        coverage_dict["LF"] = l
        pos_labels = data.loc[data[l]==1].shape[0]
        neg_labels = data.loc[data[l]==0].shape[0]
        unlab_labels = data.loc[data[l]==-1].shape[0]
        
        coverage_dict["Pos Count"] = pos_labels
        coverage_dict["Neg Count"] = neg_labels
        coverage_dict["Unlabelled Count"] = unlab_labels
        
        coverage_dict["Pos %"] = pos_labels/total_size
        coverage_dict["Neg %"] = neg_labels/total_size
        coverage_dict["Unlabelled %"] = unlab_labels/total_size
    
        coverage_list.append(coverage_dict)
    
    coverage_scores = pd.DataFrame(coverage_list)
    return coverage_scores

def get_agreements_and_conflicts(data):
    """
    """
    agreement_dict = {}
    
    agreement_data = data.loc[data["unanimous_label"] != -100]
    agreement_total_size = agreement_data.shape[0]
    
    agreement_dict["total"] = agreement_total_size
    agreement_dict["pos agreement"] = agreement_data.loc[agreement_data["unanimous_label"]==1].shape[0]
    agreement_dict["neg agreement"] = agreement_data.loc[agreement_data["unanimous_label"]==0].shape[0]
    agreement_dict["unlabelled agreement"] = agreement_data.loc[agreement_data["unanimous_label"]==-1].shape[0]
    agreement_dict["pos agreement %"] = agreement_dict["pos agreement"]/agreement_dict["total"]
    agreement_dict["neg agreement %"] = agreement_dict["neg agreement"]/agreement_dict["total"]
    agreement_dict["unlabelled agreement %"] = agreement_dict["unlabelled agreement"]/agreement_dict["total"]
    
    print(f"\nAgreement Stats for Labelling Functions :")
    print(pd.DataFrame([agreement_dict]))
    
    conflict_data = data.loc[data["unanimous_label"] == -100]
    conflict_total_size = conflict_data.shape[0]
    
    print(f"\nTotal Number of Conflicts : {conflict_total_size}")
    

def get_labelling_stats(all_train_samples):
    """
    """
    # size
    print(f"\n Total Size : {all_train_samples.shape[0]}")
    
    # label coverage
    print(f"\nLabelling Functions Coverage :")
    print(get_coverage(all_train_samples))
    
    get_agreements_and_conflicts(all_train_samples)
    
    
    
    