'''
Original Copyright (c) Facebook, Inc. and its affiliates
Modifications Copyright Ibrahim Taha Aksu, 2023.
All Rights Reserved.
'''

import argparse
version = 1.0
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T5_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--GPT_checkpoint", type=str, default="microsoft/DialoGPT-small", help="Path, url or short name of the model")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path to checkpoint directory")
    parser.add_argument("--ckpt_file", type=str, default="", help="Path to checkpoint file")
    parser.add_argument("--test_path", type=str, default="", help="Path to new test results")
    parser.add_argument("--saving_dir", type=str, default="save", help="Path for saving")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--meta_batch_size", type=int, default=1, help="Batch size for meta training")
    parser.add_argument("--dev_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id")
    parser.add_argument("--worker_number", type=int, default=16, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=5, help="Patience")
    parser.add_argument("--dataset", type=str, default="multiwoz", help="Name of the dataset to be used")

    


    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--no_early_stop", action='store_true', help="deactivate early stopping")
    parser.add_argument("--none_gate", action='store_true', help="use none gated version")
    parser.add_argument("--frozen_baseline", action='store_true', help="frozen baseline setting")
    parser.add_argument("--full_freeze", action='store_true', help="LM fully frozen setting")
    parser.add_argument("--dec_pref", action='store_true', help="prefixes are appended to decoder side as well")
    parser.add_argument("--no_freeze", action='store_true', help="don't freeze anything")
    parser.add_argument("--add_reparameterization", action='store_true', help="add reparameterization trick by the original Prefix Tuning Paper. Empriically suggested only for the SGD dataset")
    parser.add_argument("--length", type=int, default=50, help="Batch size for validation")
    parser.add_argument("--max_history", type=int, default=2, help="max number of turns in the dialogue")
    parser.add_argument("--GPU", type=int, default=1, help="number of gpu to use")
    parser.add_argument("--model_name", type=str, default="t5", help="use t5 or bart?")
    parser.add_argument("--slot_lang", type=str, default="none", help="use 'none', 'human', 'naive', 'value', 'question', 'slottype' slot description")
    parser.add_argument("--fewshot", type=float, default=0.0, help="data ratio for few shot experiment")
    parser.add_argument("--fix_label", action='store_true')
    parser.add_argument("--except_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--only_domain", type=str, default="none", help="hotel, train, restaurant, attraction, taxi")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--semi", action='store_true')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--max_size", type=int , default=500, help="Max token size of model input")

    #Model hyperparameters
    parser.add_argument("--gr_hidden_size", type=int, default=128, help="")
    parser.add_argument("--sr_hidden_size", type=int, default=128, help="")
    parser.add_argument("--mixer_hidden_size", type=int, default=128, help="")
    parser.add_argument("--nsp_hidden_size", type=int, default=32, help="")
    parser.add_argument("--se_hidden_size", type=int, default=32, help="")
    parser.add_argument("--warm_up_steps", type=int, default=1, help="")
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--val_check_interval", type=float, default=0.2)
    


    #None_gate Model hyperparameters
    parser.add_argument("--none_hidden_size", type=int, default=32, help="")
    parser.add_argument("--none_threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--harmonic_combination", action='store_true', help="combines genearation and gate losses harmonically")

    #Prompter choice
    parser.add_argument("--adapter", action='store_true', help="use adapter as prompter")
    parser.add_argument("--t5_block", action='store_true', help="use T5_block as prompter")

    # Add Adapter paramters
    parser.add_argument("--down_proj", type=int, default=64, help="") 

    parser.add_argument("--prefix_length", type=int, default=10, help="Number of prefixes to be added") 
    parser.add_argument("--prompter_dropout", type=float, default=0, help="Number of prefixes to be added") 
    parser.add_argument("--reparametrization_ratio", type=int, default=2, help="Number of prefixes to be added") 
    
    



    #Wandb parameters 
    parser.add_argument("--wandb_project_name", type=str, default="Zero_Shot_DST_T5DST_MultiWOZ_2_1_v2", help="Name of the project to be displayed in wandb UI")
    parser.add_argument("--wandb_job_type", type=str, default="train", help="Job type to be displayed in wandb")
    parser.add_argument("--wandb_run_name", type= str, default="t5_run", help = "The name of the experiments to be displayed in wandb UI")
    parser.add_argument("--wandb_group_name", type=str, default="Standard", help="Name of the experiment group to be displayed in wandb UI")
    parser.add_argument("--wandb_mode", type=str, default="online", help="Name of the experiment group to be displayed in wandb UI")
    

    args = parser.parse_args()
    return args
