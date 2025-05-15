import torch
import numpy as np
import os
import pickle as pkl
import argparse
import json
from matplotlib import pyplot as plt



if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default parameter grid search evaluation')
    parse.add_argument(
        "-cached_args_file",
        default="eval_gs_REDCLIFF_S_CMLP_tst100hzRerun1024AvgReg_BSCgsSmooth1_dataFULL_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()
    
    print("__MAIN__: LOADING ARGS", flush=True)
    with open(args.cached_args_file, 'r') as infile:
        args_dict = json.load(infile)
        trained_models_root_path = args_dict["trained_models_root_path"]
        save_root_path = args_dict["save_root_path"]
    
    selection_criteria = [
        "forecasting_loss", 
        "factor_loss", 
        "gc_cosine_sim_history", 
        "forecasting_loss_and_factor_loss_and_gc_cosine_sim_history", 
    ]
    vital_criteria = [
        "forecasting_loss", 
        "factor_loss", 
        "gc_cosine_sim_history", 
    ]
    vital_criteria_history_keys = [
        "avg_forecasting_loss", 
        "avg_factor_loss", 
        "avg_gc_factor_cos_sim_history", 
    ]

    # grab all results folders and associated .pkl summary files
    model_summary_files = {
        x:trained_models_root_path+os.sep+x+os.sep+"training_meta_data_and_hyper_parameters.pkl" for x in os.listdir(trained_models_root_path) if os.path.isfile(trained_models_root_path+os.sep+x+os.sep+"training_meta_data_and_hyper_parameters.pkl")
    }
    print("__MAIN__: unfiltered model_summary_files == ", model_summary_files, flush=True)

    # report summaries from .pkl files
    model_summaries = {key:None for key in model_summary_files.keys()}
    for i, key in enumerate(model_summary_files.keys()):
        with open(model_summary_files[key], 'rb') as infile:
            model_summaries[key] = pkl.load(infile)
            if i == 0: 
                print("__MAIN__: i==0; model_summaries[key].keys() == ", model_summaries[key].keys())
            
    print("\n__MAIN__: model_summaries LOADED")
    print("\n__MAIN__: len(model_summaries) == ", len(model_summaries), flush=True)

    # get min/max/avg/final values from roc-auc summaries for each model
    print("\n__MAIN__: min/max/avg/final values model summaries:")
    for model_key in model_summaries.keys():
        print("\t model_key == ", model_key, flush=True)
        curr_roc_auc_summary = model_summaries[model_key]["roc_auc_histories"]
        curr_roc_auc_OffDiag_summary = model_summaries[model_key]["roc_auc_OffDiag_histories"]
        curr_gc_fw_l1_penalty_summaries = model_summaries[model_key]["avg_fw_l1_penalty"]
        curr_gc_factor_l1_summaries = model_summaries[model_key]["gc_factor_l1_loss_histories"]
        curr_gc_factor_cosine_sim_histories = model_summaries[model_key]["gc_factor_cosine_sim_histories"]
        curr_gc_factor_deltacon0_histories = model_summaries[model_key]["deltacon0_histories"]
        curr_gc_factor_deltacon0_with_directed_degrees_histories = model_summaries[model_key]["deltacon0_with_directed_degrees_histories"]
        curr_gc_factor_deltaffinity_histories = model_summaries[model_key]["deltaffinity_histories"]
        curr_gc_factor_path_length_mse_histories = model_summaries[model_key]["path_length_mse_histories"]

        model_summaries[model_key]["avg_roc_auc_score_history"] = []
        model_summaries[model_key]["avg_roc_auc_OffDiag_score_history"] = []
        model_summaries[model_key]["avg_fw_l1_penalty_history"] = []
        model_summaries[model_key]["avg_gc_factor_l1_history"] = []
        model_summaries[model_key]["avg_gc_factor_cos_sim_history"] = []
        model_summaries[model_key]["avg_gc_factor_deltacon0_history"] = []
        model_summaries[model_key]["avg_gc_factor_deltacon0_with_directed_degrees_history"] = []
        model_summaries[model_key]["avg_gc_factor_deltaffinity_history"] = []
        model_summaries[model_key]["avg_gc_factor_path_length_mse_histories"] = {key:[] for key in curr_gc_factor_path_length_mse_histories.keys()}
        for factor_roc_auc_score_tuple in zip(*curr_roc_auc_summary[0.0]):
            curr_mean = np.mean(list(factor_roc_auc_score_tuple))
            model_summaries[model_key]["avg_roc_auc_score_history"].append(curr_mean)
        for factor_roc_auc_OffDiag_score_tuple in zip(*curr_roc_auc_OffDiag_summary[0.0]):
            curr_mean = np.mean(list(factor_roc_auc_OffDiag_score_tuple))
            model_summaries[model_key]["avg_roc_auc_OffDiag_score_history"].append(curr_mean)
        for fw_l1_penalty in curr_gc_fw_l1_penalty_summaries:
            model_summaries[model_key]["avg_fw_l1_penalty_history"].append(fw_l1_penalty)
        for factor_l1_norm_tuple in zip(*curr_gc_factor_l1_summaries):
            curr_mean = np.mean(list(factor_l1_norm_tuple))
            model_summaries[model_key]["avg_gc_factor_l1_history"].append(curr_mean)
        for factor_cos_sim_tuple in zip(*[curr_gc_factor_cosine_sim_histories[key] for key in curr_gc_factor_cosine_sim_histories.keys()]):
            curr_mean = np.mean(list(factor_cos_sim_tuple))
            model_summaries[model_key]["avg_gc_factor_cos_sim_history"].append(curr_mean)
        for factor_deltacon0_tuple in zip(*curr_gc_factor_deltacon0_histories):
            curr_mean = np.mean(list(factor_deltacon0_tuple))
            model_summaries[model_key]["avg_gc_factor_deltacon0_history"].append(curr_mean)
        for factor_deltacon0_with_directed_degrees_tuple in zip(*curr_gc_factor_deltacon0_with_directed_degrees_histories):
            curr_mean = np.mean(list(factor_deltacon0_with_directed_degrees_tuple))
            model_summaries[model_key]["avg_gc_factor_deltacon0_with_directed_degrees_history"].append(curr_mean)
        for factor_deltaffinity_tuple in zip(*curr_gc_factor_deltaffinity_histories):
            curr_mean = np.mean(list(factor_deltaffinity_tuple))
            model_summaries[model_key]["avg_gc_factor_deltaffinity_history"].append(curr_mean)
        for key in curr_gc_factor_path_length_mse_histories.keys():
            for curr_factor_path_length_mse_tuple in zip(*curr_gc_factor_path_length_mse_histories[key]):
                curr_mean = np.mean(list(curr_factor_path_length_mse_tuple))
                model_summaries[model_key]["avg_gc_factor_path_length_mse_histories"][key].append(curr_mean)
    
    orig_model_summary_file_names = [x for x in model_summary_files.keys()]
    for model_file_name in orig_model_summary_file_names:
        model_is_incomplete = False
        for criteria_name in vital_criteria_history_keys:
            if len(model_summaries[model_file_name][criteria_name]) == 0:
                model_is_incomplete = True
                break
            if len(model_summaries[model_file_name][criteria_name]) != len(model_summaries[model_file_name][vital_criteria_history_keys[0]]):
                model_is_incomplete = True
                break
            
        if model_is_incomplete:
            print("__MAIN__: REMOVING model ", model_file_name, " ON ACCOUNT OF MISSING DATA")
            del model_summary_files[model_file_name]
            del model_summaries[model_file_name]
        else:
            print("__MAIN__: model_file_name == ", model_file_name, " has the following necessary histories - ",len(model_summaries[model_file_name]['avg_roc_auc_score_history']),";",len(model_summaries[model_file_name]['avg_roc_auc_OffDiag_score_history']),";",len(model_summaries[model_file_name]['avg_forecasting_loss']),";",len(model_summaries[model_file_name]['avg_factor_loss']),";",len(model_summaries[model_file_name]['avg_fw_l1_penalty_history']),";",len(model_summaries[model_file_name]['avg_gc_factor_l1_history']),";",len(model_summaries[model_file_name]['avg_gc_factor_cos_sim_history']),"\n\n")
            
    print("__MAIN__: filtered model_summary_files == ", model_summary_files, flush=True)
    
    # grab stats for selection_criteria plots
    print("__MAIN__: BEGINNING SELECTION-CRITERIA EVALUATIONS", flush=True)
    criteria_performance_dict = {criteria:dict() for criteria in selection_criteria}
    for criteria_index, criteria_type in enumerate(selection_criteria):
        print("\n\n\n__MAIN__: NOW INVESTIGATING STOPPING CRITERIA TYPE criteria_type == ", criteria_type, " (", criteria_index, " / ", len(selection_criteria), ")")
        model_keys = list(model_summaries.keys())
        best_criteria_performance_by_model = []
        best_epoch_by_model = []
        if criteria_type == "roc_auc":
            best_criteria_performance_by_model = [np.max(model_summaries[model_key]['avg_roc_auc_score_history']) for model_key in model_keys]
            best_epoch_by_model = [np.argmax(model_summaries[model_key]['avg_roc_auc_score_history']) for model_key in model_keys]
        elif criteria_type == "roc_auc_OffDiag":
            best_criteria_performance_by_model = [np.max(model_summaries[model_key]['avg_roc_auc_OffDiag_score_history']) for model_key in model_keys]
            best_epoch_by_model = [np.argmax(model_summaries[model_key]['avg_roc_auc_OffDiag_score_history']) for model_key in model_keys]
        elif criteria_type == "forecasting_loss":
            best_criteria_performance_by_model = [np.min(model_summaries[model_key]['avg_forecasting_loss']) for model_key in model_keys]
            best_epoch_by_model = [np.argmin(model_summaries[model_key]['avg_forecasting_loss']) for model_key in model_keys]
        elif criteria_type == "factor_loss":
            best_criteria_performance_by_model = [np.min(model_summaries[model_key]['avg_factor_loss']) for model_key in model_keys]
            best_epoch_by_model = [np.argmin(model_summaries[model_key]['avg_factor_loss']) for model_key in model_keys]
        elif criteria_type == "fw_l1_penalty_history":
            best_criteria_performance_by_model = [np.min(model_summaries[model_key]['avg_fw_l1_penalty_history']) for model_key in model_keys]
            best_epoch_by_model = [np.argmin(model_summaries[model_key]['avg_fw_l1_penalty_history']) for model_key in model_keys]
        elif criteria_type == "gc_l1_history":
            best_criteria_performance_by_model = [np.min(model_summaries[model_key]['avg_gc_factor_l1_history']) for model_key in model_keys]
            best_epoch_by_model = [np.argmin(model_summaries[model_key]['avg_gc_factor_l1_history']) for model_key in model_keys]
        elif criteria_type == "gc_cosine_sim_history":
            best_criteria_performance_by_model = [np.min(model_summaries[model_key]['avg_gc_factor_cos_sim_history']) for model_key in model_keys]
            best_epoch_by_model = [np.argmin(model_summaries[model_key]['avg_gc_factor_cos_sim_history']) for model_key in model_keys]

        elif criteria_type == "forecasting_loss_and_factor_loss":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            for (forecast_loss_hist, factorScore_loss_hist) in zip(forecast_loss_by_model, factorScore_loss_by_model):
                combo_loss_hist = []
                for (x,y) in zip(forecast_loss_hist, factorScore_loss_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "forecasting_loss_and_fw_l1_penalty_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            fw_l1_penalty_loss_by_model = [model_summaries[model_key]['avg_fw_l1_penalty_history'] for model_key in model_keys]
            for (forecast_loss_hist, fw_l1_penalty_hist) in zip(forecast_loss_by_model, fw_l1_penalty_loss_by_model):
                combo_loss_hist = []
                for (x,y) in zip(forecast_loss_hist, fw_l1_penalty_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "forecasting_loss_and_gc_l1_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            l1_norms_by_model = [model_summaries[model_key]['avg_gc_factor_l1_history'] for model_key in model_keys]
            for (forecast_loss_hist, l1_norm_hist) in zip(forecast_loss_by_model, l1_norms_by_model):
                combo_loss_hist = []
                for (x,y) in zip(forecast_loss_hist, l1_norm_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "forecasting_loss_and_gc_cosine_sim_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            for (forecast_loss_hist, cos_sim_hist) in zip(forecast_loss_by_model, cos_sims_by_model):
                combo_loss_hist = []
                for (x,y) in zip(forecast_loss_hist, cos_sim_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "factor_loss_and_fw_l1_penalty_history":
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            fw_l1_penalty_loss_by_model = [model_summaries[model_key]['avg_fw_l1_penalty_history'] for model_key in model_keys]
            for (factorScore_loss_hist, fw_l1_penalty_hist) in zip(factorScore_loss_by_model, fw_l1_penalty_loss_by_model):
                combo_loss_hist = []
                for (x,y) in zip(factorScore_loss_hist, fw_l1_penalty_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "factor_loss_and_gc_l1_history":
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            l1_norms_by_model = [model_summaries[model_key]['avg_gc_factor_l1_history'] for model_key in model_keys]
            for (factorScore_loss_hist, l1_norm_hist) in zip(factorScore_loss_by_model, l1_norms_by_model):
                combo_loss_hist = []
                for (x,y) in zip(factorScore_loss_hist, l1_norm_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "factor_loss_and_gc_cosine_sim_history":
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            for (factorScore_loss_hist, cos_sim_hist) in zip(factorScore_loss_by_model, cos_sims_by_model):
                combo_loss_hist = []
                for (x,y) in zip(factorScore_loss_hist, cos_sim_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "gc_l1_history_and_gc_cosine_sim_history":
            l1_norms_by_model = [model_summaries[model_key]['avg_gc_factor_l1_history'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            for (l1_norm_hist, cos_sims_hist) in zip(l1_norms_by_model, cos_sims_by_model):
                combo_loss_hist = []
                for (x,y) in zip(l1_norm_hist, cos_sims_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "gc_cosine_sim_history_and_fw_l1_penalty_history":
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            fw_l1_penalty_loss_by_model = [model_summaries[model_key]['avg_fw_l1_penalty_history'] for model_key in model_keys]
            for (cos_sims_hist, fw_l1_penalty_hist) in zip(cos_sims_by_model, fw_l1_penalty_loss_by_model):
                combo_loss_hist = []
                for (x,y) in zip(cos_sims_hist, fw_l1_penalty_hist):
                    combo_loss_hist.append(x+y)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
                
        elif criteria_type == "forecasting_loss_and_factor_loss_and_fw_l1_penalty_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            fw_l1_penalty_loss_by_model = [model_summaries[model_key]['avg_fw_l1_penalty_history'] for model_key in model_keys]
            for (forecast_loss_hist, factorScore_loss_hist, fw_l1_penalty_hist) in zip(forecast_loss_by_model, factorScore_loss_by_model, fw_l1_penalty_loss_by_model):
                combo_loss_hist = []
                for (x,y,z) in zip(forecast_loss_hist, factorScore_loss_hist, fw_l1_penalty_hist):
                    combo_loss_hist.append(x+y+z)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "forecasting_loss_and_factor_loss_and_gc_l1_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            l1_norms_by_model = [model_summaries[model_key]['avg_gc_factor_l1_history'] for model_key in model_keys]
            for (forecast_loss_hist, factorScore_loss_hist, l1_norm_hist) in zip(forecast_loss_by_model, factorScore_loss_by_model, l1_norms_by_model):
                combo_loss_hist = []
                for (x,y,z) in zip(forecast_loss_hist, factorScore_loss_hist, l1_norm_hist):
                    combo_loss_hist.append(x+y+z)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "forecasting_loss_and_gc_l1_history_and_gc_cosine_sim_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            l1_norms_by_model = [model_summaries[model_key]['avg_gc_factor_l1_history'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            for (forecast_loss_hist, l1_norm_hist, cos_sims_hist) in zip(forecast_loss_by_model, l1_norms_by_model, cos_sims_by_model):
                combo_loss_hist = []
                for (x,z,c) in zip(forecast_loss_hist, l1_norm_hist, cos_sims_hist):
                    combo_loss_hist.append(x+z+c)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "forecasting_loss_and_factor_loss_and_gc_cosine_sim_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            for (forecast_loss_hist, factorScore_loss_hist, cos_sims_hist) in zip(forecast_loss_by_model, factorScore_loss_by_model, cos_sims_by_model):
                combo_loss_hist = []
                for (x,y,c) in zip(forecast_loss_hist, factorScore_loss_hist, cos_sims_hist):
                    combo_loss_hist.append(x+y+c)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "factor_loss_and_gc_l1_history_and_gc_cosine_sim_history":
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            l1_norms_by_model = [model_summaries[model_key]['avg_gc_factor_l1_history'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            for (factorScore_loss_hist, l1_norm_hist, cos_sims_hist) in zip(factorScore_loss_by_model, l1_norms_by_model, cos_sims_by_model):
                combo_loss_hist = []
                for (y,z,c) in zip(factorScore_loss_hist, l1_norm_hist, cos_sims_hist):
                    combo_loss_hist.append(y+z+c)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "forecasting_loss_and_factor_loss_and_gc_cosine_sim_history_and_fw_l1_penalty_history":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            fw_l1_penalty_loss_by_model = [model_summaries[model_key]['avg_fw_l1_penalty_history'] for model_key in model_keys]
            for (forecast_loss_hist, factorScore_loss_hist, cos_sims_hist, fw_l1_penalty_hist) in zip(forecast_loss_by_model, factorScore_loss_by_model, cos_sims_by_model, fw_l1_penalty_loss_by_model):
                combo_loss_hist = []
                for (x,y,c,z) in zip(forecast_loss_hist, factorScore_loss_hist, cos_sims_hist, fw_l1_penalty_hist):
                    combo_loss_hist.append(x+y+c+z)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        elif criteria_type == "all_criteria_combo":
            forecast_loss_by_model = [model_summaries[model_key]['avg_forecasting_loss'] for model_key in model_keys]
            factorScore_loss_by_model = [model_summaries[model_key]['avg_factor_loss'] for model_key in model_keys]
            l1_norms_by_model = [model_summaries[model_key]['avg_gc_factor_l1_history'] for model_key in model_keys]
            cos_sims_by_model = [model_summaries[model_key]['avg_gc_factor_cos_sim_history'] for model_key in model_keys]
            for (forecast_loss_hist, factorScore_loss_hist, l1_norm_hist, cos_sims_hist) in zip(forecast_loss_by_model, factorScore_loss_by_model, l1_norms_by_model, cos_sims_by_model):
                combo_loss_hist = []
                for (x,y,z,c) in zip(forecast_loss_hist, factorScore_loss_hist, l1_norm_hist, cos_sims_hist):
                    combo_loss_hist.append(x+y+z+c)
                best_criteria_performance_by_model.append(np.min(combo_loss_hist))
                best_epoch_by_model.append(np.argmin(combo_loss_hist))
        else: 
            raise NotImplementedError()
        
        print("len(model_keys) == ", len(model_keys))
        print("len(best_epoch_by_model) == ", len(best_epoch_by_model))
        print("len(model_summaries[model_key]['avg_roc_auc_score_history']) == ", len(model_summaries[model_key]["avg_roc_auc_score_history"]))
        corresponding_stopping_roc_auc_scores = [model_summaries[model_key]["avg_roc_auc_score_history"][ep] for (model_key, ep) in zip(model_keys, best_epoch_by_model)]
        corresponding_stopping_roc_auc_OffDiag_scores = [model_summaries[model_key]["avg_roc_auc_OffDiag_score_history"][ep] for (model_key, ep) in zip(model_keys, best_epoch_by_model)]
        corresponding_stopping_fw_l1_penalty = [model_summaries[model_key]["avg_fw_l1_penalty_history"][ep] for (model_key, ep) in zip(model_keys, best_epoch_by_model)]
        corresponding_stopping_avg_gc_factor_deltacon0_history = [model_summaries[model_key]["avg_gc_factor_deltacon0_history"][ep] for (model_key, ep) in zip(model_keys, best_epoch_by_model)]
        corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history = [
            model_summaries[model_key]["avg_gc_factor_deltacon0_with_directed_degrees_history"][ep] for (model_key, ep) in zip(model_keys, best_epoch_by_model)
        ]
        corresponding_stopping_avg_gc_factor_deltaffinity_history = [model_summaries[model_key]["avg_gc_factor_deltaffinity_history"][ep] for (model_key, ep) in zip(model_keys, best_epoch_by_model)]
        corresponding_stopping_avg_gc_factor_path_length_mse_histories = []
        for (model_key, ep) in zip(model_keys, best_epoch_by_model):
            curr_dict = {mse_key: None for mse_key in model_summaries[model_key]["avg_gc_factor_path_length_mse_histories"].keys()}
            for key in model_summaries[model_key]["avg_gc_factor_path_length_mse_histories"].keys():
                if len(model_summaries[model_key]["avg_gc_factor_path_length_mse_histories"][key]) > ep:
                    curr_dict[key] = model_summaries[model_key]["avg_gc_factor_path_length_mse_histories"][key][ep]
                else:
                    curr_dict[key] = None
            corresponding_stopping_avg_gc_factor_path_length_mse_histories.append(curr_dict)

        criteria_performance_dict[criteria_type] = {
            "model_keys": model_keys, 
            "best_criteria_performance_by_model": best_criteria_performance_by_model, 
            "best_epoch_by_model": best_epoch_by_model, 
            "corresponding_stopping_roc_auc_scores": corresponding_stopping_roc_auc_scores, 
            "corresponding_stopping_roc_auc_OffDiag_scores": corresponding_stopping_roc_auc_OffDiag_scores, 
            "corresponding_stopping_fw_l1_penalty": corresponding_stopping_fw_l1_penalty, 
            "corresponding_stopping_avg_gc_factor_deltacon0_history": corresponding_stopping_avg_gc_factor_deltacon0_history, 
            "corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history": corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history, 
            "corresponding_stopping_avg_gc_factor_deltaffinity_history": corresponding_stopping_avg_gc_factor_deltaffinity_history, 
            "corresponding_stopping_avg_gc_factor_path_length_mse_histories": corresponding_stopping_avg_gc_factor_path_length_mse_histories, 
        }
        print("__MAIN__: \t - model_keys == ", model_keys)
        print("__MAIN__: \t - best_criteria_performance_by_model == ", best_criteria_performance_by_model)
        print("__MAIN__: \t - best_epoch_by_model == ", best_epoch_by_model)
        print("__MAIN__: \t - corresponding_stopping_roc_auc_scores == ", corresponding_stopping_roc_auc_scores)
        print("__MAIN__: \t - corresponding_stopping_roc_auc_OffDiag_scores == ", corresponding_stopping_roc_auc_OffDiag_scores)
        print("__MAIN__: \t - corresponding_stopping_fw_l1_penalty == ", corresponding_stopping_fw_l1_penalty)
        print("__MAIN__: \t - corresponding_stopping_avg_gc_factor_deltacon0_history == ", corresponding_stopping_avg_gc_factor_deltacon0_history)
        print("__MAIN__: \t - corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history == ", corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history)
        print("__MAIN__: \t - corresponding_stopping_avg_gc_factor_deltaffinity_history == ", corresponding_stopping_avg_gc_factor_deltaffinity_history)
        print("__MAIN__: \t - corresponding_stopping_avg_gc_factor_path_length_mse_histories == ", corresponding_stopping_avg_gc_factor_path_length_mse_histories)

        finite_performances = []
        for x in best_criteria_performance_by_model:
            if np.isfinite(x):
                finite_performances.append(x)
            else:
                finite_performances.append(np.inf)
        selected_model_from_criteria = None
        if criteria_type != "roc_auc" and criteria_type != "roc_auc_OffDiag":
            selected_model_from_criteria = np.argmin(finite_performances)
        else:
            selected_model_from_criteria = np.argmax(finite_performances)
        selected_model_from_criteria_path = model_keys[selected_model_from_criteria]
        selected_model_from_criteria_performance = best_criteria_performance_by_model[selected_model_from_criteria]
        selected_model_from_criteria_best_epoch = best_epoch_by_model[selected_model_from_criteria]
        selected_model_from_criteria_stopping_roc_auc_score = corresponding_stopping_roc_auc_scores[selected_model_from_criteria]
        selected_model_from_criteria_stopping_roc_auc_OffDiag_score = corresponding_stopping_roc_auc_OffDiag_scores[selected_model_from_criteria]
        selected_model_from_criteria_stopping_fw_l1_penalty = corresponding_stopping_fw_l1_penalty[selected_model_from_criteria]
        selected_model_from_criteria_stopping_avg_gc_factor_deltacon0_similarity = corresponding_stopping_avg_gc_factor_deltacon0_history[selected_model_from_criteria]
        selected_model_from_criteria_stopping_avg_gc_factor_deltacon0_with_directed_degrees_similarity = corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history[selected_model_from_criteria]
        selected_model_from_criteria_stopping_avg_gc_factor_deltaffinity_similarity = corresponding_stopping_avg_gc_factor_deltaffinity_history[selected_model_from_criteria]
        selected_model_from_criteria_stopping_avg_gc_factor_path_length_mse = corresponding_stopping_avg_gc_factor_path_length_mse_histories[selected_model_from_criteria]
        print("\n\n__MAIN__: <<< BEST MODEL BASED ON CRITERIA == "+criteria_type+" >>> ----------------------------------------------------------------")
        print("__MAIN__: selected_model_from_criteria == ", selected_model_from_criteria)
        print("__MAIN__: selected_model_from_criteria_path == ", selected_model_from_criteria_path)
        print("__MAIN__: selected_model_from_criteria_performance == ", selected_model_from_criteria_performance)
        print("__MAIN__: selected_model_from_criteria_best_epoch == ", selected_model_from_criteria_best_epoch)
        print("__MAIN__: selected_model_from_criteria_stopping_roc_auc_score == ", selected_model_from_criteria_stopping_roc_auc_score)
        print("__MAIN__: selected_model_from_criteria_stopping_roc_auc_OffDiag_score == ", selected_model_from_criteria_stopping_roc_auc_OffDiag_score)
        print("__MAIN__: selected_model_from_criteria_stopping_fw_l1_penalty == ", selected_model_from_criteria_stopping_fw_l1_penalty)
        print("__MAIN__: selected_model_from_criteria_stopping_avg_gc_factor_deltacon0_similarity == ", selected_model_from_criteria_stopping_avg_gc_factor_deltacon0_similarity)
        print("__MAIN__: selected_model_from_criteria_stopping_avg_gc_factor_deltacon0_with_directed_degrees_similarity == ", selected_model_from_criteria_stopping_avg_gc_factor_deltacon0_with_directed_degrees_similarity)
        print("__MAIN__: selected_model_from_criteria_stopping_avg_gc_factor_deltaffinity_similarity == ", selected_model_from_criteria_stopping_avg_gc_factor_deltaffinity_similarity)
        print("__MAIN__: selected_model_from_criteria_stopping_avg_gc_factor_path_length_mse == ", selected_model_from_criteria_stopping_avg_gc_factor_path_length_mse)
        print("__MAIN__: -----------------------------------------------------------------------------------------------------------------------------------------------\n\n", flush=True)

        pearson_r_epoch_perf_avg = np.corrcoef(best_criteria_performance_by_model, best_epoch_by_model)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(best_epoch_by_model, best_criteria_performance_by_model, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("best epoch")
        ax1.set_ylabel("best "+criteria_type)
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_epoch_perf_avg))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_performance_by_epoch_scatter_vis.png")
        plt.close()

        pearson_r_rocAuc_avg = np.corrcoef(best_criteria_performance_by_model, corresponding_stopping_roc_auc_scores)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(best_criteria_performance_by_model, corresponding_stopping_roc_auc_scores, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("best "+criteria_type)
        ax1.set_ylabel("corresponding avg. ROC-AUC score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_rocAuc_avg))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_against_rocAuc_avg_scatter_vis.png")
        plt.close()

        ZOOM_XLIM_LOWER_BOUND = np.min(best_criteria_performance_by_model)-1
        ZOOM_XLIM_UPPER_BOUND = ZOOM_XLIM_LOWER_BOUND + (np.max(best_criteria_performance_by_model)-ZOOM_XLIM_LOWER_BOUND)*0.2
        locally_best_criteria_performance = []
        locally_relevant_roc_auc_scores = []
        for (x,y) in zip(best_criteria_performance_by_model, corresponding_stopping_roc_auc_scores):
            if x >= ZOOM_XLIM_LOWER_BOUND and x <= ZOOM_XLIM_UPPER_BOUND:
                locally_best_criteria_performance.append(x)
                locally_relevant_roc_auc_scores.append(y)

        pearson_r_rocAuc_avg_ZOOMED = np.corrcoef(locally_best_criteria_performance, locally_relevant_roc_auc_scores)[0,1]
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(locally_best_criteria_performance, locally_relevant_roc_auc_scores, alpha=0.5)
        ax1.set_xlabel("best "+criteria_type)
        ax1.set_ylabel("corresponding avg. ROC-AUC score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", LOCAL Pearson's R="+str(pearson_r_rocAuc_avg_ZOOMED))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_against_rocAuc_avg_scatter_vis_ZOOMED.png")
        plt.close()
        
        pearson_r_rocAucOffDiag_avg = np.corrcoef(best_criteria_performance_by_model, corresponding_stopping_roc_auc_OffDiag_scores)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(best_criteria_performance_by_model, corresponding_stopping_roc_auc_OffDiag_scores, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("best "+criteria_type)
        ax1.set_ylabel("corresponding avg. ROC-AUC OffDiag score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_rocAucOffDiag_avg))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_against_rocAucOffDiag_avg_scatter_vis.png")
        plt.close()

        ZOOM_XLIM_LOWER_BOUND = np.min(best_criteria_performance_by_model)-1
        ZOOM_XLIM_UPPER_BOUND = ZOOM_XLIM_LOWER_BOUND + (np.max(best_criteria_performance_by_model)-ZOOM_XLIM_LOWER_BOUND)*0.2
        locally_best_criteria_performance = []
        locally_relevant_roc_auc_OffDiag_scores = []
        for (x,y) in zip(best_criteria_performance_by_model, corresponding_stopping_roc_auc_OffDiag_scores):
            if x >= ZOOM_XLIM_LOWER_BOUND and x <= ZOOM_XLIM_UPPER_BOUND:
                locally_best_criteria_performance.append(x)
                locally_relevant_roc_auc_OffDiag_scores.append(y)

        pearson_r_rocAucOffDiag_avg_ZOOMED = np.corrcoef(locally_best_criteria_performance, locally_relevant_roc_auc_OffDiag_scores)[0,1]
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(locally_best_criteria_performance, locally_relevant_roc_auc_OffDiag_scores, alpha=0.5)
        ax1.set_xlabel("best "+criteria_type)
        ax1.set_ylabel("corresponding avg. ROC-AUC OffDiag score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", LOCAL Pearson's R="+str(pearson_r_rocAucOffDiag_avg_ZOOMED))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_against_rocAucOffDiag_avg_scatter_vis_ZOOMED.png")
        plt.close()
        
        pearson_r_rocAuc_avg_by_deltaCon0 = np.corrcoef(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_roc_auc_scores)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_roc_auc_scores, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("corresponding avg. DeltaCon0 similarity score")
        ax1.set_ylabel("corresponding avg. ROC-AUC score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_rocAuc_avg_by_deltaCon0))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_rocAuc_avg_by_deltaCon0_avg_scatter_vis.png")
        plt.close()
        
        pearson_r_rocAucOffDiag_avg_by_deltaCon0 = np.corrcoef(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_roc_auc_OffDiag_scores)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_roc_auc_OffDiag_scores, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("corresponding avg. DeltaCon0 similarity score")
        ax1.set_ylabel("corresponding avg. ROC-AUC OffDiag score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_rocAucOffDiag_avg_by_deltaCon0))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_rocAucOffDiag_avg_by_deltaCon0_avg_scatter_vis.png")
        plt.close()
        
        pearson_r_DDdeltaCon0_avg_by_deltaCon0 = np.corrcoef(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_avg_gc_factor_deltacon0_with_directed_degrees_history, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("corresponding avg. DeltaCon0 similarity score")
        ax1.set_ylabel("orresponding avg. DeltaCon0 withDirectedDegrees similarity score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_DDdeltaCon0_avg_by_deltaCon0))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_deltaCon0withDirectedDegrees_avg_by_deltaCon0_avg_scatter_vis.png")
        plt.close()
        
        pearson_r_deltafinity_avg_by_deltaCon0 = np.corrcoef(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_avg_gc_factor_deltaffinity_history)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(corresponding_stopping_avg_gc_factor_deltacon0_history, corresponding_stopping_avg_gc_factor_deltaffinity_history, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("corresponding avg. DeltaCon0 similarity score")
        ax1.set_ylabel("corresponding avg. Deltafinity score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_deltafinity_avg_by_deltaCon0))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_deltafinity_avg_by_deltaCon0_avg_scatter_vis.png")
        plt.close()

        pearson_r_deltaCon0_avg_by_epoch = np.corrcoef(best_epoch_by_model, corresponding_stopping_avg_gc_factor_deltacon0_history)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(best_epoch_by_model, corresponding_stopping_avg_gc_factor_deltacon0_history, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("best epoch")
        ax1.set_ylabel("corresponding avg. DeltaCon0 score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_deltaCon0_avg_by_epoch))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_deltaCon0_avg_by_epoch_scatter_vis.png")
        plt.close()

        # plot comparissons of 10 best models' forecasting and factor score mse, the sum of both, and their deltaCon0's 
        pearson_r_deltaCon0_avg = np.corrcoef(best_criteria_performance_by_model, corresponding_stopping_avg_gc_factor_deltacon0_history)[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(best_criteria_performance_by_model, corresponding_stopping_avg_gc_factor_deltacon0_history, c=best_epoch_by_model, alpha=0.5)
        ax1.set_xlabel("best "+criteria_type)
        ax1.set_ylabel("corresponding avg. DeltaCon0 score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", Pearson's R="+str(pearson_r_deltaCon0_avg))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_against_deltaCon0_avg_scatter_vis.png")
        plt.close()

        ZOOM_XLIM_LOWER_BOUND = np.min(best_criteria_performance_by_model)-1
        ZOOM_XLIM_UPPER_BOUND = ZOOM_XLIM_LOWER_BOUND + (np.max(best_criteria_performance_by_model)-ZOOM_XLIM_LOWER_BOUND)*0.2
        locally_best_criteria_performance = []
        locally_relevant_deltaCon0_scores = []
        for (x,y) in zip(best_criteria_performance_by_model, corresponding_stopping_avg_gc_factor_deltacon0_history):
            if x >= ZOOM_XLIM_LOWER_BOUND and x <= ZOOM_XLIM_UPPER_BOUND:
                locally_best_criteria_performance.append(x)
                locally_relevant_deltaCon0_scores.append(y)

        pearson_r_deltaCon0_avg_ZOOMED = np.corrcoef(locally_best_criteria_performance, locally_relevant_deltaCon0_scores)[0,1]
        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(12.8, 9.6)
        ax1.scatter(locally_best_criteria_performance, locally_relevant_deltaCon0_scores, alpha=0.5)
        ax1.set_xlabel("best "+criteria_type)
        ax1.set_ylabel("corresponding avg. DeltaCon0 score")
        ax1.set_title("Stopping Criteria: "+criteria_type+", LOCAL Pearson's R="+str(pearson_r_deltaCon0_avg_ZOOMED))
        plt.legend()
        plt.tight_layout()
        plt.draw()
        fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+criteria_type+"_against_deltaCon0_avg_scatter_vis_ZOOMED.png")
        plt.close()
    
    for i, c1_key in enumerate(criteria_performance_dict.keys()):
        for j, c2_key in enumerate(criteria_performance_dict.keys()):
            if i < j:
                assert c1_key != c2_key # sanity check
                curr_pearson_r_score = np.corrcoef(criteria_performance_dict[c1_key]["best_criteria_performance_by_model"], criteria_performance_dict[c2_key]["best_criteria_performance_by_model"])[0,1] # see https://realpython.com/numpy-scipy-pandas-correlation-python/ and REDACTED FOR ANONYMITY
                fig1, ax1 = plt.subplots()
                fig1.set_size_inches(12.8, 9.6)
                ax1.scatter(criteria_performance_dict[c1_key]["best_criteria_performance_by_model"], criteria_performance_dict[c2_key]["best_criteria_performance_by_model"], alpha=0.5)
                ax1.set_xlabel("best "+c1_key)
                ax1.set_ylabel("best "+c2_key)
                ax1.set_title("Stopping Criteria Correlation: Pearson's R="+str(curr_pearson_r_score))
                plt.legend()
                plt.tight_layout()
                plt.draw()
                fig1.savefig(save_root_path+os.sep+"stopping_criteria_"+c1_key+"_vs_"+c2_key+"_scatter_vis.png")
                plt.close()

    
    print("__MAIN__: DONE!!!")
    pass
