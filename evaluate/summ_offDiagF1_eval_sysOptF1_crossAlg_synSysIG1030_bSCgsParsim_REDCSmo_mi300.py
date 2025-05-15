import os
import json
import shutil
import argparse

if __name__=="__main__":
    parse=argparse.ArgumentParser(description='Default parameter system-level evaluation')
    parse.add_argument(
        "-cached_args_file",
        default="summ_offDiagF1_eval_sysOptF1_crossAlg_synSysIG1030_bSCgsParsim_REDCSmo_mi300_cached_args.txt", # WARNING: edit this line when running new experiments on SLURM server
        help="/path/to/cached_args.txt should contain a json object with updated arg values",
    )
    args = parse.parse_args()
    
    print("__MAIN__: LOADING ARGS", flush=True)
    root_paths_to_trained_models = []
    with open(args.cached_args_file, 'r') as infile:
        args_dict = json.load(infile)
    
    cv_eval_folders = [args_dict["orig_eval_root_path"]+os.sep+x for x in os.listdir(args_dict["orig_eval_root_path"]) if "data_set_vis_" not in x]
    for cv_eval_folder in cv_eval_folders:
        eval_vis_folder = [x for x in os.listdir(cv_eval_folder) if x[:2] == "cv"]
        if len(eval_vis_folder) == 1: 
            eval_vis_folder = eval_vis_folder[0]
            sys_cv_name = eval_vis_folder[:-1*len("_edgesNonlinear_labelsOneHot_noiT-gaussian_noiL-1-0_oFscF_data")]
            if "factor_level_key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag_f1_vals_across_factors_by_algorithm.png" in os.listdir(cv_eval_folder+os.sep+eval_vis_folder):
                shutil.copy(cv_eval_folder+os.sep+eval_vis_folder+os.sep+"factor_level_key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag_f1_vals_across_factors_by_algorithm.png", 
                    args_dict["save_root_path"]
                )
                shutil.move(args_dict["save_root_path"]+os.sep+"factor_level_key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag_f1_vals_across_factors_by_algorithm.png", 
                    args_dict["save_root_path"]+os.sep+sys_cv_name+"_factor_level_estGC_normOffDiag_vs_trueGC_normOffDiag_f1_vals_across_factors_by_alg.png"
                )
            if "REDCLIFF_S_CMLP_WithSmoothing_IMPROVEMENTS" in os.listdir(cv_eval_folder+os.sep+eval_vis_folder):
                shutil.copy(cv_eval_folder+os.sep+eval_vis_folder+os.sep+"REDCLIFF_S_CMLP_WithSmoothing_IMPROVEMENTS"+os.sep+"factor_level_key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag_f1_vals_across_factors_by_algorithm.png", 
                    args_dict["save_root_path"]
                )
                shutil.move(args_dict["save_root_path"]+os.sep+"factor_level_key_stats_estGC_normOffDiag_vs_trueGC_normOffDiag_f1_vals_across_factors_by_algorithm.png", 
                    args_dict["save_root_path"]+os.sep+sys_cv_name+"_factor_level_estGC_normOffDiag_vs_trueGC_normOffDiag_f1_REDCLIFFSCMLPWithSmoothingPairwiseIMPROVEMENTS_across_factors.png"
                )
                
    print("__MAIN__: DONE !!!")
    pass