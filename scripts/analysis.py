from ast import Gt
from distutils import core
import random
import json
import copy
import sys
from this import d
random.seed(10)
line_sep = "-------------------------------------------\n"
section_sep = "===================================\n\
===================================\n"

class Model_Result():
    def __init__(self,all_turn_beliefs, all_preds_beliefs,wrong_preds,total_turn, total_pred,total_ground_truth,wrong_pred,precision, recall, F1 ,missed_label,missed_label_ratio,over_pred_label,over_pred_label_ratio,none_pred_acc,model_name="unnamed" ) -> None:
        self.all_turn_beliefs = all_turn_beliefs
        self.none_pred_acc = none_pred_acc
        self.all_preds_beliefs = all_preds_beliefs
        self.wrong_preds = wrong_preds
        self.total_turn = total_turn
        self.name = model_name
        self.total_pred = total_pred
        self.total_ground_truth = total_ground_truth
        self.wrong_pred = wrong_pred
        self.precision = precision
        self.recall = recall 
        self.F1 = F1
        self.missed_label = missed_label
        self.missed_label_ratio = missed_label_ratio
        self.over_pred_label = over_pred_label
        self.over_pred_label_ratio=over_pred_label_ratio

    def textify(self):
        result = ("Statistics for model @ {}:\n\
{}\
Total number of predictions: {}\n\
Total number of ground truth values: {}\n\
Total number of wrong predictions: {} (P: {:.2f}, R: {:.2f}, F1: {:.2f})\n\
Total number of missed labels: {} ({:.2f}%)\n\
Total number of over-predictions: {} ({:.2f}%)\n\
None prediction accuracy: {:.2f}%\n\
-------------------------------------------\n").format(self.name,line_sep,self.total_pred,self.total_ground_truth,self.wrong_pred,self.precision, self.recall, self.F1 ,self.missed_label,self.missed_label_ratio,self.over_pred_label,self.over_pred_label_ratio,self.none_pred_acc)
    # print (result)
        return result


def intersect_wrong_preds(mr1,mr2):
    wp1 = mr1.wrong_preds
    wp2 = mr2.wrong_preds
    final_wp = {}
    for dial_key in wp1.keys():
        for turn_id in wp1[dial_key]:
            if dial_key in wp2.keys():
                if turn_id in wp2[dial_key]:
                    if dial_key not in final_wp.keys():
                        final_wp[dial_key] = []
                    final_wp[dial_key].append(turn_id)
                else:
                    continue
            else:
                continue
    return final_wp


def get_belief_dict(belief):
    belief_dict = {}
    for el in belief:
        label = "-".join(el.split("-")[:2])
        value = el.split("-")[-1]

        belief_dict[label] = value
    return belief_dict

def generate_stats(dataPath, data, N):
    total_pred, total_ground_truth , wrong_pred , correct_pred , missed_label , over_pred_label,total_turn = 0 ,0 ,0 ,0 ,0 , 0 , 0
    wrong_preds = {}
    all_preds_beliefs = {}
    all_turn_beliefs = {}

    for dial_key in data.keys():
        turns = data[dial_key]["turns"]
        total_turn += len(turns)
        for turn_id in turns.keys():
            turn_belief = get_belief_dict(turns[turn_id]["turn_belief"])
            pred_belief = get_belief_dict(turns[turn_id]["pred_belief"])

            if dial_key not in all_turn_beliefs.keys():
                all_turn_beliefs[dial_key]={}
                all_preds_beliefs[dial_key]={}

            all_turn_beliefs[dial_key][turn_id]=turn_belief
            all_preds_beliefs[dial_key][turn_id]=pred_belief


            if turn_belief != pred_belief:
                #JA -1 
                if dial_key not in wrong_preds:
                    wrong_preds[dial_key] = []
                wrong_preds[dial_key].append(turn_id)
            total_ground_truth += len(turn_belief.keys())
            total_pred += len(pred_belief.keys())

            for label in turn_belief.keys():
                if label in pred_belief:
                    if pred_belief[label] == turn_belief[label]:
                        correct_pred += 1
                    else:
                        wrong_pred +=1
                else:
                    missed_label +=1
            
            for label in pred_belief.keys():
                if label not in turn_belief:
                    over_pred_label += 1
    none_pred_acc = (total_turn * N - missed_label - over_pred_label) / (total_turn * N) * 100
    #TODO: N being the total slot label count for that domain. i.e. 4 for taxi domain

    precision = correct_pred / total_pred
    recall = correct_pred / total_ground_truth
    F1 = 2 * precision * recall / (precision + recall)
    missed_label_ratio = (missed_label / total_ground_truth) * 100
    over_pred_label_ratio = (over_pred_label / total_pred) * 100

    mr = Model_Result(all_turn_beliefs, all_preds_beliefs, wrong_preds,total_turn, total_pred,total_ground_truth,wrong_pred,precision, recall,F1 ,missed_label,missed_label_ratio,over_pred_label,over_pred_label_ratio,none_pred_acc)
    return mr 
def get_conf_matr(mr1,mr2):
    #             mr2
    #           T    F
    # mr1   T 
    #       F
    TT, TF, FT , FF = 0,0,0,0
    TFs = {}
    FTs = {}
    for wrong_pred_dial in mr1.wrong_preds.keys():
        if wrong_pred_dial not in mr2.wrong_preds.keys():
            #this whole dialogue was correct by mr2, add everything here to FT
            FT += len(mr1.wrong_preds[wrong_pred_dial])
            if wrong_pred_dial not in FTs:
                FTs[wrong_pred_dial] = []
            for turn_id in mr1.wrong_preds[wrong_pred_dial]:
                FTs[wrong_pred_dial].append(turn_id)
        else:
            for turn_id in mr1.wrong_preds[wrong_pred_dial]:
                if turn_id in mr2.wrong_preds[wrong_pred_dial]:
                    #both did this turn wrong
                    FF += 1
                else:
                    #only m1 did this turn wrong
                    FT += 1
                    if wrong_pred_dial not in FTs:
                        FTs[wrong_pred_dial] = []
                    FTs[wrong_pred_dial].append(turn_id)
                
            for turn_id in mr2.wrong_preds[wrong_pred_dial]:
                if turn_id in mr1.wrong_preds[wrong_pred_dial]:
                    pass # FFs are already processed above
                else:
                    #only m2 did this turn wrong
                    TF += 1
                    if wrong_pred_dial not in TFs:
                        TFs[wrong_pred_dial] = []
                    TFs[wrong_pred_dial].append(turn_id)

    for wrong_pred_dial in mr2.wrong_preds.keys():
        if wrong_pred_dial not in mr1.wrong_preds.keys():
            TF += len(mr2.wrong_preds[wrong_pred_dial])
            if wrong_pred_dial not in TFs:
                TFs[wrong_pred_dial] = []
            for turn_id in mr2.wrong_preds[wrong_pred_dial]:
                TFs[wrong_pred_dial].append(turn_id)
        #else case already adressed above (where they both have mistakes in the dialogue)

    # The rest of the turns are TT
    TT = mr1.total_turn - TF - FT - FF
    
    conf_mat ="\t\t{}\n\t\t\t\tT\t\tF\n{}\tT\t{}\t{}\n\t\t\tF\t{}\t\t{}\n".format(mr2.name,mr1.name,TT,TF,FT,FF)

    assert TF == sum([len(L) for L in TFs.values()])
    assert FT == sum([len(L) for L in FTs.values()])

    return TFs, FTs, conf_mat , TT, TF, FT , FF
def get_dial_hist(dial_id, turn):
    dh = "Dialogue ID: {} (Until turn: {})\n".format(dial_id,turn)
    test_data_path = "/home/users/astar/i2r/stuait/scratch/Projects/disentangled_zero_dst/T5DST/data/test_dials.json"
    test_data = json.load(open(test_data_path,"r"))

    for dial in test_data:
        if dial["dial_id"] != dial_id:
            continue

        for i in range(int(turn)+1):
            dh += "System: {}\n".format(dial["turns"][i]["system"])
            dh += "User: {}\n".format(dial["turns"][i]["user"])
        break
    return dh
def sample_from_list(samples,mr,count,result):
    for i in range(count):
        result+= line_sep
        dial, turns = random.choice(list(samples.items()))
        turn = random.choice(turns)
        samples[dial].remove(turn)
        if not samples[dial]:
            del samples[dial]
        
        dh = get_dial_hist(dial, turn)
        result += "Dialogue History:\n{}\n".format(dh)
        result += "Ground truth turn predictions:\n{}\n".format(mr.all_turn_beliefs[dial][turn])
        result+= "Model {}, predictions:\n{}\n".format(mr.name, mr.all_preds_beliefs[dial][turn])
    return result

def specific_filter(TFs, FTs, slot_names):
    test_data_path = "/home/users/astar/i2r/stuait/scratch/Projects/disentangled_zero_dst/T5DST/data/test_dials.json"
    test_data = json.load(open(test_data_path,"r"))
    test_data = {item["dial_id"]:item for item in test_data}
    
    conf_mats = [copy.deepcopy(TFs),copy.deepcopy(FTs)]
    for conf_mat in conf_mats:
        to_remove_dials = set()
        for dial_id,turn_ids in conf_mat.items():
            to_remove_turns = set()
            for turn_id in turn_ids:
                
                slots = list(test_data[dial_id]["turns"][int(turn_id)]["state"]["slot_values"].keys())
                for name in slot_names:
                    if name not in slots:
                        to_remove_turns.add(turn_id)
                        break
                        # conf_mat[dial_id].remove(turn_id)
            for remove_turn in to_remove_turns:
                conf_mat[dial_id].remove(remove_turn)
            if not conf_mat[dial_id]:
                to_remove_dials.add(dial_id)
        for remove_dial in to_remove_dials:
            del conf_mat[remove_dial]
    return conf_mats[0],conf_mats[1]
    
    
    
def sample_dialogues(TFs, FTs,mr1,mr2, count,slot_names=None):
    result = ""
    if slot_names:
        TFs ,FTs  = specific_filter (TFs, FTs,slot_names)
    if count == "-1":
        TF_count = len(TFs)
        FT_count = len(FTs)
    else:
        TF_count = min(count,sum([len(L) for L in TFs.values()]))
        FT_count = min(count,sum([len(L) for L in FTs.values()]))

    result += "{} True false samples:\n".format(TF_count)
    result = sample_from_list(TFs,mr2,TF_count,result)
    result += line_sep

    result += "{} False True samples:\n".format(FT_count)
    result = sample_from_list(FTs,mr1,FT_count,result)
    result += line_sep

    return result


def main():
    model_names = sys.argv[1:3]
    predictions = sys.argv[3:5]
    result = sys.argv[5]
    domain = sys.argv[6]
    sample_count = int(sys.argv[7])
    slot_names = sys.argv[8:]
    if not slot_names:
        slot_names = None
    f = open("{}/results_{}_{}_vs_{}.txt".format(result,domain,model_names[0],model_names[1]), "w")
    model_results = []
    
    N=None
    slot_counts = {"train": 6,
                   "restaurant": 7,
                   "hotel":10,
                   "taxi":4,
                   "attraction":3}
    N = slot_counts[domain]

    for model_name,prediction in zip(model_names,predictions):
        data = json.load(open(prediction,'r'))
        mr = generate_stats(prediction,data, N)
        mr.name = model_name
        model_results.append(mr)
        
    f.write("{} Domain Results:\n".format(domain))

    for mr in model_results:
        f.write(mr.textify())

    f.write(section_sep)
    f.write("Confusion Matrix (across turns))\n")

    '''
    TODO

    1. Create and draw confusion matrices between the models
    
    2.  Then choose random turns that show discrepancy between two models.
    (M1 False, M2 True). 

        a. Print the subject turn and its history in a nice format.
        b. Print the wrong prediction by M1
    
    Repeat step 2 for M2,M1 respectively.
    '''
    #HACK model results indices seem random, how do you choose

    TFs, FTs, conf_mat ,_ ,_ ,_ ,_ = get_conf_matr(model_results[0],model_results[1])
    f.write (conf_mat)
    f.write(section_sep)
    f.write("Sample TFs and FTs \n")
    

    result = sample_dialogues(TFs, FTs,model_results[0],model_results[1],sample_count,slot_names)
    f.write(result)
    f.write(section_sep)
    print("done!")
    f.close()
    

if __name__ == "__main__":
    main()