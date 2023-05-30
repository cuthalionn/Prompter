import json,os
import collections
CATEGORIES = ["#ALL_SERVICES","#SEEN_SERVICES","#UNSEEN_SERVICES","Buses","Messaging","Payment","Trains","Events","Flights","Homes","Media","Movies","Music","RentalCars"]
# METRICS = ["average_goal_accuracy","joint_goal_accuracy"]
METRICS = ["joint_goal_accuracy"]
class Solution:

    def __init__(self,paths):
        self.paths = paths
        self.load_data()
        self.aggregate_data()
        
    
    def load_data(self):
        self.data = {}
        for p in self.paths:
            datum = json.load(open(os.path.join(p,"results","script_result","result.json"),"r"))
            self.data[p] = datum
    
    def aggregate_data(self):
        self.agg_dict = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
        for p,datum in self.data.items():
            for cat,its_dict in datum.items():
                if cat not in CATEGORIES:
                    continue
                for k,item in its_dict.items():
                    if k not in METRICS:
                        continue
                    self.agg_dict[cat][k].append(item)
        
        for cat, its_dict in self.agg_dict.items():
            for k,item in its_dict.items():
                self.agg_dict[cat][k].append(sum(self.agg_dict[cat][k]) / (len(self.agg_dict[cat][k])))
    
    def __str__(self):
        str_result = []
        str_result.append("*"*20)
        for dic_key in self.agg_dict:
            str_result.append(f"{dic_key}:")
            for metric,value in self.agg_dict[dic_key].items():
                str_result.append(f"{metric} : {value}")
        str_result.append("*"*20)
        return "\n".join(str_result)


#BASELINE PATHS
# fb_paths = ["PATH_1_TO_MODEL_DIR", "PATH_2_TO_MODEL_DIR", "PATH_3_TO_MODEL_DIR"]
fb_paths = ["/home/users/astar/i2r/stuait/scratch/Projects/disentangled_zero_dst/T5DST/src/save/t5REP_FOR_CR_SGD_pluto_global_local_prompt_do_0.2_p10_rept_noes_2ep_249_except_domain_none_slotlang_none_lr_0.0001_epoch_2_seed_249"]
#MODEL PATHS
# p_paths = ["PATH_1_TO_MODEL_DIR", "PATH_2_TO_MODEL_DIR", "PATH_3_TO_MODEL_DIR"]
p_paths = ["/home/users/astar/i2r/stuait/scratch/Projects/disentangled_zero_dst/T5DST/src/save/t5REP_FOR_CR_SGD_pluto_global_local_prompt_do_0.2_p10_rept_noes_2ep_249_except_domain_none_slotlang_none_lr_0.0001_epoch_2_seed_249"]

sol_fb = Solution(fb_paths)
print(sol_fb)
# print("*"*20)

sol_p = Solution(p_paths)
print(sol_p)