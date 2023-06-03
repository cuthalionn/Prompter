'''
Original Copyright Ibrahim Taha Aksu, 2023.
All Rights Reserved.
'''

"""
Load 5 different models each missing training on an individual domain `{D}`.
Then initialize the key and value prefixes of each layer `{(N,6,2,512)}` for each slot description (30) and store them. 
Then find cosine similarity for prefix key and values between `{D}`'s slots and other domains' slots and plot a heatmap.
Eventually you should get 5 heatmaps showing the similarities of prefixes for each training settings.

"""
from src.T5_prefix_tuning import DST_Prefixed
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from src.config import get_args
from src.prefix_data_loader import prepare_data, get_slot_information
import torch
from pytorch_lightning import seed_everything
import os 
import json
from torch import nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

class Solution():

    def __init__(self,args):
        self.ontology = json.load(open("../data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))

        self.args = args

    def prepare_inputs(self):
        
        # Load the model and get prefix values
        tokenizer = T5TokenizerFast.from_pretrained("t5-small",
                                                    bos_token="[bos]",
                                                    eos_token="[eos]",
                                                    sep_token="[sep]")

        _, _, _,_, global_prompts = prepare_data(self.args, tokenizer)
        self.task = DST_Prefixed.load_from_checkpoint(self.args["ckpt_file"],common_tokens=global_prompts).cuda()

        self.all_slots = sorted(get_slot_information(self.ontology),key = lambda x: x.split("-")[0])
        description = json.load(open("../src/utils/slot_description.json", 'r'))
        self.descriptions = []
        for slot in self.all_slots:
            self.descriptions.append(description[slot][self.args["slot_lang"]])

        tokenizer_out = self.task.tokenizer(self.descriptions,padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)
        slot_description = tokenizer_out["input_ids"]
        attentions = tokenizer_out["attention_mask"]
        key,value = self.task.get_prefixes(slot_description.cuda(),attentions.cuda())
        
        # Key
        layer_averaged = torch.mean(key,0)
        token_averaged = torch.mean(layer_averaged,1)
        reshaped_key = torch.reshape(token_averaged,(30,512))

        # Value
        layer_averaged = torch.mean(value,0)
        token_averaged = torch.mean(layer_averaged,1)
        reshaped_value = torch.reshape(token_averaged,(30,512))
        
        return reshaped_key, reshaped_value

    def calculate_cos_matrix(self,cos_matrix):

        except_slots = []
        source_slots = []
        for slot in self.all_slots:
            if slot.split("-")[0] == self.args["except_domain"]:
                except_slots.append("-".join([ slot.split("-")[0][:2] , slot.split("-")[1] ]))
            else:
                source_slots.append("-".join([ slot.split("-")[0][:2] , slot.split("-")[1] ]))

        cos_sims = []
        cos = nn.CosineSimilarity(dim=0)
        for i in range(cos_matrix.shape[0]):
            slot = self.all_slots[i]
            slot_domain = slot.split("-")[0]
            if slot_domain != self.args["except_domain"]:
                continue
            for j in range(cos_matrix.shape[0]):
                slot2 = self.all_slots[j]
                slot_domain2 = slot2.split("-")[0]
                if not slot_domain2 == self.args["except_domain"]:
                    out = cos(cos_matrix[i],cos_matrix[j])
                    cos_sims.append(out)

        # Normalize
        cos_sims = torch.Tensor(cos_sims).reshape(len(except_slots),len(source_slots))
        cos_sims -= cos_sims.min(0, keepdim=True)[0]
        cos_sims /= cos_sims.max(0, keepdim=True)[0]    
        
        return cos_sims,source_slots,except_slots

    def draw_heat_map(self,cos_sims, source_slots, except_slots,keys):
        except_idcs = range(len(except_slots))
        source_idcs = range(len(source_slots))
        # Draw a heatmap showing the similarities of prefixes for each training setting.
        cos_sims = cos_sims.cpu().detach().numpy()
        # print(f"cos_sims_shape:{cos_sims.shape}, source_slots_shape: {len(source_slots)}, except_slots_shape:{len(except_slots)}")
        cos_sims = cos_sims[except_idcs,:].reshape(len(except_idcs),len(source_slots))
        # print(f"cos_sims_shape:{cos_sims.shape}, source_slots_shape: {len(source_slots)}, except_slots_shape:{len(except_slots)}")
        cos_sims = cos_sims[:,source_idcs].reshape(len(except_idcs),len(source_idcs))
        except_slots = [except_slots[i] for i in except_idcs]
        source_slots = [source_slots[i] for i in source_idcs]
        # print(f"cos_sims_shape:{cos_sims.shape}, source_slots_shape: {len(source_slots)}, except_slots_shape:{len(except_slots)}")
        fig, ax = plt.subplots(figsize=(10,10))  # Sample figsize in inches

        # Change dark color to high similarity
        cmap = sns.cm.rocket_r

        sns.heatmap(cos_sims, 
                        linewidths=.5, 
                        # annot=True,
                        ax=ax, 
                        xticklabels=source_slots, 
                        yticklabels=except_slots, 
                        square = True,
                        cmap = cmap,
                        cbar=False
                        )
        keyword = "keys" if keys else "values"
        domain = self.args["except_domain"]
        plt.yticks(rotation=45) 
        plt.savefig(f"../figures/heatmap_{domain}_{keyword}.png",bbox_inches='tight')

    def draw_tsne(self,reps,keys=True):

        tokens = reps.cpu().detach().numpy()
        labels = self.all_slots

            
        tsne_model = TSNE(perplexity=25, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        cdict = {"train":"red","taxi":"orange","attraction":"yellow","hotel":"brown","restaurant":"blue"}
        plt.figure(figsize=(16, 16)) 
        for i in range(len(x)):
            color = cdict[labels[i].split("-")[0]]
            plt.scatter(x[i],y[i],c=color)
            plt.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')

        keyword = "keys" if keys else "values"
        domain = self.args["except_domain"]
        plt.savefig(f"../figures/tsne_{domain}_{keyword}.png")
        
    def draw_pca(self,vectors,keys=True):
        vectors = vectors.cpu().detach().numpy()
        labels = self.all_slots
        # perform PCA on the list of vectors
        pca = PCA(n_components=2)
        pca.fit(vectors)
        transformed_vectors = pca.transform(vectors)

        # create a figure to save the visualization
        fig = plt.figure()

        # plot the transformed vectors
        cdict = {"train":"red","taxi":"orange","attraction":"green","hotel":"brown","restaurant":"blue"}
        
        
        for i, vector in enumerate(transformed_vectors):
            color = cdict[labels[i].split("-")[0]]
            plt.scatter(transformed_vectors[i, 0], transformed_vectors[i, 1], c=color)
            plt.annotate(f"{labels[i]}", vector,ha="center", va="bottom")
            
        # save the figure
        keyword = "keys" if keys else "values"
        domain = self.args["except_domain"]
        fig.savefig(f"../figures/pca_{domain}_{keyword}.png")
        

    def list_most_similar_slots(self, cos_sims, source_slots, except_slots, k,keys=True):

        _, idx = torch.sort(cos_sims,dim=1,descending=True)
        idx = idx[:,:5].tolist()

        similar_slots = [[source_slots[idx[j][i]] for i in range(k)] for j in range(len(except_slots))]
        key_word = "key" if keys else "value"
        for i,slot in enumerate(except_slots):

            print(f"Most similar {k} {key_word} prefixes to {slot} are: {similar_slots[i]}")
        

    def run(self):

        key_reps, value_reps = self.prepare_inputs()

        cos_sims, source_slots, except_slots= self.calculate_cos_matrix(key_reps)
        self.draw_heat_map(cos_sims, source_slots, except_slots,keys=True)
        self.list_most_similar_slots(cos_sims, source_slots, except_slots,5,keys=True)
        # self.draw_tsne(key_reps,keys=True)
        self.draw_pca(key_reps,keys=True)
        
        

        cos_sims, source_slots, except_slots= self.calculate_cos_matrix(value_reps)
        self.draw_heat_map(cos_sims, source_slots, except_slots,keys=False)
        self.list_most_similar_slots(cos_sims, source_slots, except_slots,5,keys=False)
        # self.draw_tsne(value_reps, keys=False)
        self.draw_pca(value_reps,keys=False)


def main():
    args = get_args() 
    args = vars(args)
    seed_everything(args["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpu_id"])
    
    sol = Solution(args)
    sol.run()

if __name__ == "__main__":
    main()