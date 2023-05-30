'''
Original Copyright (c) Facebook, Inc. and its affiliates
Modifications Copyright Ibrahim Taha Aksu, 2023.
All Rights Reserved.
'''

from optparse import Values
import os
from transformers import (AdamW, BartTokenizer,
                        #   T5ForConditionalGeneration,
                          BartForConditionalGeneration)
from src.modelling_t5_copy import T5ForConditionalGeneration
# from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5Block
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from src.prompters import Prefix_Prompter
from src.prefix_data_loader import prepare_data
from src.config import get_args
from src.evaluate import evaluate_metrics
import json
from tqdm import tqdm

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

class DST_Prefixed(pl.LightningModule):
    
    def __init__(self, args, tokenizer, model, common_tokens=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]
        args["model_config"] = model.config
        self.fb = args["frozen_baseline"]
        self.ff = args["full_freeze"]
        self.decoder_prefix = args["dec_pref"] if "dec_pref" in args else False
        self.no_freeze = args["no_freeze"] if "no_freeze" in args else False
        self.train_bs = args["train_batch_size"]
        self.dev_bs = args["dev_batch_size"]
        self.test_bs = args["test_batch_size"]
        self.prompter_dropout = args["prompter_dropout"] if "prompter_dropout" in args else 0.0
        self.prefix_prompter = Prefix_Prompter(args, model,args["down_proj"],self.model.config.num_layers,self.prompter_dropout)
        self.warm_up_steps = args["warm_up_steps"]
        self.phase = 1
        self.prefix_length = args["prefix_length"] if "prefix_length" in args else 500
        # self.prefix_length = 500  # Some runs accidentally overwrote the prefix length argument use this line for those. E.g. attr fb
        self.mode = args["mode"]
        self.common_tokens = common_tokens if common_tokens is not None else "a" * 200
        self.save_hyperparameters()
        self.add_reparameterization = args["add_reparameterization"]

        self.phase1_checked = False
        self.phase2_checked = False
        
        self.global_prompts_set = False
        self.final_global_prompt = None
        if not self.add_reparameterization:
            self.final_global_prompt = torch.nn.Parameter(torch.zeros((self.prefix_length,self.model.config.d_model)))
        else:
            self.global_prompt = torch.nn.Parameter(data = torch.rand(self.prefix_length,model.config.d_model // args["reparametrization_ratio"]), # Default reparametrization ratio is 2
                                                    requires_grad = False )
            self.reparametrizer = nn.Linear(in_features = model.config.d_model // args["reparametrization_ratio"],
                                            out_features = model.config.d_model)        
            self.final_global_prompt = torch.nn.Parameter(data = torch.zeros(self.prefix_length,model.config.d_model),
                                                requires_grad = False )
        
        if self.ff:
            print("T5 model fully frozen")
            for param in self.model.parameters():
                param.requires_grad = False
    
    def init_global_prompt(self):

        assert not self.global_prompts_set

        initial_prompt = " ".join(pair[0] for pair in self.common_tokens)
        global_prompt_tokens = self.tokenizer(initial_prompt,padding=True, return_tensors="pt", add_special_tokens=False, verbose=False)["input_ids"][0][:self.prefix_length].cuda()
        global_prompt_data = torch.squeeze(self.model.shared(global_prompt_tokens),0)
        self.final_global_prompt.data = (global_prompt_data)

        self.global_prompts_set = True

    def get_attentions(self,input,input_attentions, desc, desc_attention):
        self.model.eval()
        self.prefix_prompter.eval()
        prefix_key = None
        prefix_value = None
        # New Code Start
        encoder_input = input.cuda()
        encoder_attn = input_attentions.cuda()
        
        placeholder_out = self.tokenizer("out text", padding=True, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)['input_ids'].cuda()
        
        
        prompt_embed = self.model.shared(desc.cuda())
        expanded_prompt = torch.unsqueeze(self.final_global_prompt,1).expand(-1,1,-1)
        prefix_key, prefix_value = self.prefix_prompter(expanded_prompt.cuda(),
                                                        prompt_embed,
                                                        desc_attention.cuda(),
                                                        input.shape[0])
        prefix_key = prefix_key.cuda()
        prefix_value = prefix_value.cuda()
        # New Code End 

        dec_prefix_key = None
        dec_prefix_value = None
        if self.decoder_prefix:
            dec_prefix_key = prefix_key
            dec_prefix_value = prefix_value

        model_output = self.model(input_ids = encoder_input,
                                  attention_mask = encoder_attn,
                                  labels = placeholder_out,
                                  encoder_prefix_key = prefix_key,
                                  encoder_prefix_value = prefix_value,
                                  decoder_prefix_key = dec_prefix_key,
                                  decoder_prefix_value = dec_prefix_value,
                                  output_attentions = True,
                                  return_dict = True
                                  )
        
        return model_output["encoder_attentions"] # Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
    
    def get_prefixes(self,slot_descriptions,attentions):
        prompt_embed = self.model.shared(slot_descriptions)
        expanded_prompt = torch.unsqueeze(self.final_global_prompt,1).expand(-1,len(slot_descriptions),-1)
        prefix_key, prefix_value = self.prefix_prompter(expanded_prompt.cuda(),
                                                        prompt_embed,
                                                        attentions.cuda(),
                                                        len(slot_descriptions))
        return prefix_key,prefix_value

    def generate(self, batch):
        self.model.eval().cuda()
        self.prefix_prompter.eval().cuda()
        prefix_key = None
        prefix_value = None
        # New Code Start
        encoder_input = batch["fb_encoder_input"].cuda()
        encoder_attn = batch["fb_encoder_attn_mask"].cuda()

        if not self.fb:
            prompt_embed = self.model.shared(batch["slot_desc_input"].cuda())
            expanded_prompt = torch.unsqueeze(self.final_global_prompt,1).expand(-1,self.test_bs,-1)
            prefix_key, prefix_value = self.prefix_prompter(expanded_prompt.cuda(),
                                                            prompt_embed,
                                                            batch["slot_desc_attn_mask"].cuda(),
                                                            batch["encoder_input"].shape[0])
            prefix_key = prefix_key.cuda()
            prefix_value = prefix_value.cuda()
        # New Code End

        eos_token_id = self.tokenizer.eos_token_id

        dec_prefix_key = None
        dec_prefix_value = None
        if self.decoder_prefix:
            dec_prefix_key = prefix_key
            dec_prefix_value = prefix_value


        dst_outputs = self.model.generate(input_ids=encoder_input,
                                          attention_mask=encoder_attn,
                                          eos_token_id=eos_token_id,
                                          max_length=200,
                                          encoder_prefix_key = prefix_key,
                                          encoder_prefix_value = prefix_value,
                                          decoder_prefix_key = dec_prefix_key,
                                          decoder_prefix_value = dec_prefix_value
                                          )
        return dst_outputs

    def training_step(self, batch, batch_idx):
        
        if not self.add_reparameterization and not self.global_prompts_set:
            self.init_global_prompt()
        
        # Need to add optimizer_idx parameter if using optimizer_step function
        #self.freeze_test()
        self.model.train()
        self.prefix_prompter.train()
        prefix_key = None
        prefix_value = None
        # New Code Start
        encoder_input = batch["fb_encoder_input"].cuda()
        encoder_attn = batch["fb_encoder_attn_mask"].cuda()
        
        if not self.fb:
            final_global_prompt = self.final_global_prompt
            if self.add_reparameterization:
                final_global_prompt = self.reparametrizer(self.global_prompt)
            prompt_embed = self.model.shared(batch["slot_desc_input"].cuda())
            expanded_prompt = torch.unsqueeze(final_global_prompt,1).expand(-1,self.train_bs,-1)
            prefix_key, prefix_value = self.prefix_prompter(expanded_prompt.cuda(),
                                                            prompt_embed,
                                                            batch["slot_desc_attn_mask"].cuda(),
                                                            batch["encoder_input"].shape[0])
            prefix_key = prefix_key.cuda()
            prefix_value = prefix_value.cuda()
        # New Code End 

        dec_prefix_key = None
        dec_prefix_value = None
        if self.decoder_prefix:
            dec_prefix_key = prefix_key
            dec_prefix_value = prefix_value

        model_output = self.model(input_ids = encoder_input,
                                  attention_mask = encoder_attn,
                                  labels = batch["decoder_output"],
                                  encoder_prefix_key = prefix_key,
                                  encoder_prefix_value = prefix_value,
                                  decoder_prefix_key = dec_prefix_key,
                                  decoder_prefix_value = dec_prefix_value
                                  )
        loss = model_output["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # def training_step_end(self, training_step_outputs):
    #     self.log("train_loss",training_step_outputs.sum(), on_step=True, on_epoch=False,prog_bar=True,sync_dist=True)
    #     return {'loss': training_step_outputs.sum()}

    def validation_step(self, batch, batch_idx):
        #self.freeze_test()
        self.model.eval()
        self.prefix_prompter.eval().cuda()  
        prefix_key = None
        prefix_value = None
        # New Code Start
        encoder_input = batch["fb_encoder_input"].cuda()
        encoder_attn = batch["fb_encoder_attn_mask"].cuda()
        if not self.fb:
            final_global_prompt = self.final_global_prompt
            if self.add_reparameterization:
                final_global_prompt = self.reparametrizer(self.global_prompt)
            self.final_global_prompt.data = final_global_prompt
            prompt_embed = self.model.shared(batch["slot_desc_input"].cuda())
            expanded_prompt = torch.unsqueeze(final_global_prompt,1).expand(-1,self.dev_bs,-1)
            prefix_key, prefix_value = self.prefix_prompter(expanded_prompt.cuda(),
                                                            prompt_embed,
                                                            batch["slot_desc_attn_mask"].cuda(),
                                                            batch["encoder_input"].shape[0])
            prefix_key = prefix_key.cuda()
            prefix_value = prefix_value.cuda()
        # New Code End

        dec_prefix_key = None
        dec_prefix_value = None
        if self.decoder_prefix:
            dec_prefix_key = prefix_key
            dec_prefix_value = prefix_value

        model_output = self.model(input_ids = encoder_input,
                                  attention_mask = encoder_attn,
                                  labels = batch["decoder_output"],
                                  encoder_prefix_key = prefix_key,
                                  encoder_prefix_value = prefix_value,
                                  decoder_prefix_key = dec_prefix_key,
                                  decoder_prefix_value = dec_prefix_value
                                  )

        loss = model_output["loss"]
        self.log("val_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def freeze_test(self):
        if self.mode == "train":
            if self.ff and not self.phase1_checked:
                for param in self.model.parameters():
                    assert param.requires_grad == False, "Model parameters frozen despite the option"
                self.phase1_checked = True
            if not self.ff and self.phase==1 and not self.phase1_checked:
                for param in self.model.parameters():
                    assert param.requires_grad == True, "Model parameters are not frozen despite the option"
                self.phase1_checked = True
            if not self.ff and self.phase==2 and not self.phase2_checked:
                for param in self.model.encoder.block[-1].parameters():
                    assert  param.requires_grad == True, "Model parameters frozen despite the option"
                for param in self.model.encoder.block[-3].parameters():
                    assert param.requires_grad == False, "Model parameters not frozen despite the option"
                self.phase2_checked = True
        return None

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        using_native_amp,
        using_lbfgs,
    ):
        if self.ff:
            optimizer.step(closure=optimizer_closure)
        else:

            # warm up lr
            if self.phase == 1:
                if self.global_step >= self.warm_up_steps and not self.no_freeze:
                    self.phase = 2
                    print("Second phase start")

                    # Unfreeze setting
                    
                    for param in self.model.parameters():
                        param.requires_grad = False

                    for param in self.model.encoder.block[0].parameters():
                        param.requires_grad = True

                    for param in self.model.encoder.block[-1].parameters():
                        param.requires_grad = True

                    for param in self.model.decoder.block[0].parameters():
                        param.requires_grad = True

                    for param in self.model.decoder.block[-1].parameters():
                        param.requires_grad = True

                    for param in self.model.lm_head.parameters():
                        param.requires_grad = True

                # If optimizer_idx == 0:
                optimizer.step(closure=optimizer_closure)
            else:
                # update params
                # if optimizer_idx == 1:
                optimizer.step(closure=optimizer_closure)

    # def configure_optimizers(self):
    #     return AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,correct_bias=True),AdamW(self.prompt_layer.parameters(), lr=self.lr,correct_bias=True)

    def configure_optimizers(self):
        return AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,correct_bias=True)




class DST_Seq2Seq(pl.LightningModule):

    def __init__(self, args, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.lr = args["lr"]


    def training_step(self, batch, batch_idx):
        self.model.train()
        # ORIGINAL CODE
        model_output = self.model(input_ids=batch["encoder_input"],
                                attention_mask=batch["encoder_attention_mask"],
                                labels=batch["decoder_output"]
                                )
        # ORIGINAL CODE
        # With padding  
        # (loss), *_ = self.model(input_ids=batch["se_input"],
        #                     attention_mask=batch["se_attention_mask"],
        #                     lm_labels=batch["decoder_output"]
        #                     )
        # With padding 
        self.log("train_loss",
                 model_output["loss"],
                 on_step=True,
                 on_epoch=False,
                 prog_bar=True)
               
        return {'loss': model_output["loss"], 'log': {'train_loss': model_output["loss"]}}

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        model_output = self.model(input_ids=batch["encoder_input"],
                                attention_mask=batch["encoder_attention_mask"],
                                labels=batch["decoder_output"]
                                )

        self.log("val_loss", model_output["loss"], on_step=True, on_epoch=False, prog_bar=True)

        return {'val_loss': model_output["loss"], 'log': {'val_loss': model_output["loss"]}}

    def validation_epoch_end(self, outputs):
        val_loss_mean = sum([o['val_loss'] for o in outputs]) / len(outputs)
        # show val_loss in progress bar but only log val_loss
        results = {'progress_bar': {'val_loss': val_loss_mean.item()}, 'log': {'val_loss': val_loss_mean.item()},
                   'val_loss': val_loss_mean.item()}
        return results

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, correct_bias=True)


def do_inference(args, *more):
    args = vars(args) 
    run_name = args["wandb_run_name"]
    wandb_logger = WandbLogger(name=run_name,
                               project=args["wandb_project_name"],
                               job_type=args["wandb_job_type"],
                               group=args["wandb_group_name"]) 

    args["model_name"] = args["model_name"] + args["wandb_run_name"] + \
        "_except_domain_"+args["except_domain"] + "_slotlang_" + \
        str(args["slot_lang"]) + "_lr_" + str(args["lr"]) + "_epoch_" + \
        str(args["n_epochs"]) + "_seed_" + str(args["seed"])
    run_name = args["wandb_run_name"]
    wandb_logger = WandbLogger(name= run_name,project=args["wandb_project_name"],job_type=args["wandb_job_type"] ,group=args["wandb_group_name"]) 
    # train!
    seed_everything(args["seed"])
    
    tokenizer = T5TokenizerFast.from_pretrained(args["T5_checkpoint"],
                                                    bos_token="[bos]",
                                                    eos_token="[eos]",
                                                    sep_token="[sep]")
    model = T5ForConditionalGeneration.from_pretrained(args["T5_checkpoint"])
    tokenizer = T5TokenizerFast.from_pretrained(args["T5_checkpoint"],
                                                bos_token="[bos]",
                                                eos_token="[eos]",
                                                sep_token="[sep]")
    model.resize_token_embeddings(new_num_tokens=len(tokenizer))
    _, _, test_loader, ALL_SLOTS, _, _, _ = prepare_data(args, tokenizer)

    task = None

    # TODO calculate and insert these weights
    # to args inside to data_loader module
    args["pos_none_weight_train"] = pos_none_weight_train
    args["pos_none_weight_val"] = pos_none_weight_val

    tokenizer = T5TokenizerFast.from_pretrained(args["model_checkpoint"],
                                                bos_token="[bos]",
                                                eos_token="[eos]",
                                                sep_token="[sep]")
    if args["use_baseline"]:
        task = DST_Seq2Seq(args, tokenizer, model)
    else:    
        task = DST_Prompted(args, tokenizer, model)

    if args["ckpt_file"] == "":
        task.load_state_dict(torch.load("{}/task.pt".format(args["model_checkpoint"])))
    else:
        task = task.load_from_checkpoint("{}/{}".format(args["model_checkpoint"], args["ckpt_file"]))
               
    task.eval()
    print("test start...")
    # Evaluate model
    _ = evaluate_model(args, task, task.tokenizer, task.model,
                       test_loader, args["test_path"],
                       ALL_SLOTS, wandb_logger)


def train(args, *more):
    args = vars(args)

    args["model_name"] = args["model_name"] + args["wandb_run_name"] + \
        "_except_domain_"+args["except_domain"] + "_slotlang_" +\
        str(args["slot_lang"]) + "_lr_" + str(args["lr"]) + "_epoch_" + \
        str(args["n_epochs"]) + "_seed_" + str(args["seed"])

    run_name = args["wandb_run_name"]
    wandb_logger = WandbLogger(name= run_name,project=args["wandb_project_name"],job_type=args["wandb_job_type"] ,group=args["wandb_group_name"]) 

    # train!
    seed_everything(args["seed"])

    tokenizer = None
    if "t5" in args["model_name"]:
        model = T5ForConditionalGeneration.from_pretrained(args["T5_checkpoint"])
        tokenizer = T5TokenizerFast.from_pretrained(args["T5_checkpoint"],
                                                    bos_token="[bos]",
                                                    eos_token="[eos]",
                                                    sep_token="[sep]")
    elif "bart" in args["model_name"]:
        model = BartForConditionalGeneration.from_pretrained(args["T5_checkpoint"])
        tokenizer = BartTokenizer.from_pretrained(args["T5_checkpoint"],
                                                  bos_token="[bos]",
                                                  eos_token="[eos]",
                                                  sep_token="[sep]")

        model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    train_loader, val_loader, test_loader,all_slots,global_prompts = prepare_data(args, tokenizer)
    

    task = None
    if args["use_baseline"]:
        task = DST_Seq2Seq(args, tokenizer, model)
    else:
        task = DST_Prefixed(args, tokenizer, model,global_prompts)

    if args["model_checkpoint"] != "":
        task.load_state_dict(torch.load("{}/task.pt".format(args["model_checkpoint"])))


    #save model path
    save_path = os.path.join(args["saving_dir"],args["model_name"])
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss',filepath=save_path,mode="min")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint_callback = ModelCheckpoint(filepath= save_path+"/{epoch}-{global_step}-{val_loss:.2f}",monitor='val_loss',verbose = False,save_last= True,save_top_k = 1, mode="min",)
    callbacks = []
    if not args["no_early_stop"]:
        callbacks= [pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=args["min_delta"], patience=args["patience"],verbose=False, mode='min')]
    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=callbacks,
                    checkpoint_callback = checkpoint_callback,
                    gpus=args["GPU"],
                    deterministic=True, 
                    num_nodes=1,
                    #precision=16,
                    accelerator="ddp",
                    logger = wandb_logger,
                    val_check_interval = 0.20,
                    )

    trainer.fit(task, train_loader, val_loader)
    # if args["use_baseline"]:
    #     task = DST_Seq2Seq.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    # else:    
    #     task = DST_Prompted.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    task = task.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(task.state_dict(), "{}/task.pt".format(save_path))
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    _ = evaluate_model(args, task, task.tokenizer, task.model, test_loader, save_path,all_slots, wandb_logger)

def evaluate_model(args,task, tokenizer, model, test_loader, save_path, ALL_SLOTS,wandb_logger, prefix="zeroshot",):

    save_path = os.path.join(save_path,"results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    predictions = {}
    # to gpu
    # gpu = args["GPU"][0]
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    slot_logger = {slot_name:[0,0,0] for slot_name in ALL_SLOTS}

    for batch in tqdm(test_loader):
        if args["use_baseline"]:
            dst_outputs = model.generate(input_ids=batch["encoder_input"].to(device),
                                    attention_mask=batch["attention_mask"].to(device),
                                    eos_token_id=tokenizer.eos_token_id,
                                    max_length=200,
                                    )
        else:
            dst_outputs = task.generate(batch)
        
        value_batch = tokenizer.batch_decode(dst_outputs, skip_special_tokens=True)

        for idx, value in enumerate(value_batch):
            #For some reason the new generation adds a trailing whitespace to each decoded value 
            #This is my naive solution.
            value = value.strip()
 
            dial_id = batch["ID"][idx]
            if dial_id not in predictions:
                predictions[dial_id] = {}
                predictions[dial_id]["domain"] = batch["domains"][idx][0]
                predictions[dial_id]["turns"] = {}
            if batch["turn_id"][idx] not in predictions[dial_id]["turns"]:
                predictions[dial_id]["turns"][batch["turn_id"][idx]] = {"turn_belief":batch["turn_belief"][idx], "pred_belief":[]}
                
            if args["dataset"] == "sgd":
                pred_slot = str(batch["domain"][idx])+ '-' + str(batch["slot_text"][idx])
            else:
                pred_slot = str(batch["slot_text"][idx])
                
            if value!="none":
                predictions[dial_id]["turns"][batch["turn_id"][idx]]["pred_belief"].append(pred_slot+ '-' + str(value))
            
            # analyze slot acc:
            if str(value)==str(batch["value_text"][idx]):
                slot_logger[pred_slot][1]+=1 # hit
            slot_logger[pred_slot][0]+=1 # total
            
            

    for slot_log in slot_logger.values():
        slot_log[2] = (slot_log[1]/slot_log[0]) if slot_log[0] != 0 else 0

    with open(os.path.join(save_path, f"{prefix}_slot_acc.json"), 'w') as f:
        json.dump(slot_logger,f, indent=4)

    with open(os.path.join(save_path, f"{prefix}_prediction.json"), 'w') as f:
        json.dump(predictions,f, indent=4)

    joint_acc_score, F1_score, turn_acc_score = evaluate_metrics(predictions, ALL_SLOTS)

    evaluation_metrics = {"Joint Acc":joint_acc_score, "Turn Acc":turn_acc_score, "Joint F1":F1_score}
    slot_acc_metrics = {key:value[2] for (key,value) in slot_logger.items()}
    wandb_logger.log_metrics(metrics= slot_acc_metrics)
    wandb_logger.log_metrics(metrics= evaluation_metrics)
    
    print(f"{prefix} result:",evaluation_metrics)

    with open(os.path.join(save_path, f"{prefix}_result.json"), 'w') as f:
        json.dump(evaluation_metrics,f, indent=4)

    return predictions


def fine_tune(args, *more):
    args = vars(args)
    
    domains = ["hotel", "train", "restaurant", "attraction", "taxi"]
    for domain in domains:
        if domain in args["T5_checkpoint"]:
            args["only_domain"] = domain
            
    assert args["only_domain"]!="none", "only_domain is not indicated"
    assert args["except_domain"]=="none", "except_domain should be set to none for finetuning"

    args["model_name"] = args["model_name"] + args["wandb_run_name"] + \
        "_only_domain_"+args["only_domain"] + "_few_shot_"+ str(args["fewshot"]) + "_slotlang_" +\
        str(args["slot_lang"]) + "_lr_" + str(args["lr"]) + "_epoch_" + \
        str(args["n_epochs"]) + "_seed_" + str(args["seed"])

    run_name = args["wandb_run_name"]
    wandb_logger = WandbLogger(name= run_name,project=args["wandb_project_name"],job_type=args["wandb_job_type"] ,group=args["wandb_group_name"]) 

    seed_everything(args["seed"])
    assert (args["model_checkpoint"] != ""), "no checkpoint file is provided"

    loaded_task = DST_Prefixed.load_from_checkpoint(args["model_checkpoint"],strict=False,).cuda()
    tokenizer = loaded_task.tokenizer

    train_loader, val_loader, test_loader, \
        ALL_SLOTS, fewshot_loader_dev, fewshot_loader_test,global_prompts = prepare_data(args, tokenizer)

    task = None
    if args["use_baseline"]:
        task = DST_Seq2Seq(args, tokenizer, loaded_task.model)
    else:
        task = DST_Prefixed(args, tokenizer, loaded_task.model, global_prompts)

    
    # HACK Ensure that this also updates the tokenizer.
    # task.load_state_dict(torch.load("{}/task.pt".format(args["model_checkpoint"])))

    task.load_state_dict(loaded_task.state_dict(),strict=False)
    task.global_prompts_set = True # HACK This was being thrown when init method is called during finetuning.
    save_path = os.path.join(args["saving_dir"],args["model_name"])

    wandb_logger = WandbLogger(name= run_name,project=args["wandb_project_name"],job_type=args["wandb_job_type"] ,group=args["wandb_group_name"]) 

    checkpoint_callback = ModelCheckpoint(filepath= save_path+"/{epoch}-{global_step}-{val_loss:.2f}",
    monitor='val_loss',verbose = False,save_last= True,save_top_k = 1, mode="min",)
    callbacks = []
    if not args["no_early_stop"]:
        callbacks= [pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=args["min_delta"], patience=args["patience"],verbose=False, mode='min')]

    trainer = Trainer(
                    default_root_dir=save_path,
                    accumulate_grad_batches=args["gradient_accumulation_steps"],
                    gradient_clip_val=args["max_norm"],
                    max_epochs=args["n_epochs"],
                    callbacks=callbacks,
                    checkpoint_callback = checkpoint_callback,
                    gpus=args["GPU"],
                    deterministic=True,
                    num_nodes=1,
                    # precision=16,
                    accelerator="dp",
                    logger = wandb_logger,
                    val_check_interval = args["val_check_interval"]
                    )
    
    trainer.fit(task, train_loader, val_loader)

    task = task.load_from_checkpoint(checkpoint_callback.best_model_path)
    task.global_prompts_set = True
    torch.save(task.state_dict(), "{}/task.pt".format(save_path))
    task.tokenizer.save_pretrained(save_path)

    print("test start...")
    #evaluate model
    ratio = "ratio_" + str(args["fewshot"]) + "_seed_" + str(args["seed"])
    _ = evaluate_model(args,task, task.tokenizer, task.model, test_loader,save_path, ALL_SLOTS,wandb_logger, prefix=ratio)

def main():

    args = get_args() 
    os.environ['WANDB_MODE'] = args.wandb_mode
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    #Preliminary Checks
    assert args.dataset in ["multiwoz", "sgd"], "{} is not a valid dataset name".format(args.dataset)
    
    
    if args.mode=="train":
        train(args)
    if args.mode=="finetune":
        fine_tune(args)
    if args.mode=="infer":
        do_inference(args)

if __name__ == "__main__":
    main()
