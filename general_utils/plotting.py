import torch
import numpy as np
from matplotlib import pyplot as plt
import os
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import gridspec
import random

from general_utils.misc import flatten_GC_estimate_with_lags



def plot_cross_experiment_summary(save_path, alg_performance_means_tOpt, alg_performance_sems_tOpt, alg_names, dataset_names, mean_colors, sem_colors, 
                                  title, #='Synthetic System Edge Prediction'
                                  ylabel, #="Synthetic System Name ("+r'$n_c$'+"-"+r'$n_e$'+"-"+r'$n_k$'+")", 
                                  xlabel, #='Avg. Optimal F1-Score '+r'$\pm$'+' Std. Err. of the Mean', 
                                  bar_width=0.9, fig_width=9, fig_height=7, FONT_SMALL_SIZE=18, FONT_MEDIUM_SIZE=20, FONT_BIGGER_SIZE=22, x_domain_lim=None, sys_name_includes_snr_label=False):
    def get_data_name_alias_for_plot_axes(orig_name):
        return orig_name + "   "

    def get_data_name_alias(orig_name):
        split_data_name = None
        if sys_name_includes_snr_label:
            split_data_name = orig_name[5:].split("_")
        else:
            split_data_name = orig_name.split("_")
        abrieve_split_data_name = [int(x[2:]) for x in split_data_name]
        data_name_alias = "-".join([str(x) for x in abrieve_split_data_name])
        if sys_name_includes_snr_label:
            data_name_alias = orig_name[:4]+"-"+data_name_alias
        return data_name_alias

    plt.rc('font', size=FONT_SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=FONT_BIGGER_SIZE)  # fontsize of the figure title
    
    # remainder of code drafted with help from ChatGPT
    # Create figure and axis
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0])
    
    index = np.arange((len(alg_names)+1)*len(dataset_names))
    total_num_bars_per_dset = len(alg_names)
    width_of_bars_per_dset = bar_width

    alg_names = alg_names + [None]
    mean_colors = mean_colors + [None]
    sem_colors = sem_colors + [None]
    alg_index_offset = []
    for d in range(len(dataset_names)):
        alg_index_offset = alg_index_offset + [d for _ in range(len(alg_names))]

    # Horizontal bar plot with whiskers
    for a, (alg_name, mean_color, sem_color) in enumerate(zip(alg_names, mean_colors, sem_colors)):
        if alg_name is not None:
            curr_inds = None
            curr_means = None
            curr_sems = None
            curr_inds = [ind for ind in index if ind % len(alg_names) == a]
            curr_means = [alg_performance_means_tOpt[ind-offset] for ind, offset in zip(index, alg_index_offset) if ind % len(alg_names) == a]
            curr_sems = [alg_performance_sems_tOpt[ind-offset] for ind, offset in zip(index, alg_index_offset) if ind % len(alg_names) == a]
            ax1.barh([ind - width_of_bars_per_dset/2 for ind in curr_inds], curr_means, xerr=curr_sems, ecolor=sem_color, height=bar_width, color=mean_color, capsize=5, label=alg_name)
    
    ax1.set_yticks([i-1 for i in index])
    ylabels = []
    for ind, offset in zip(index, alg_index_offset):
        if ind % len(alg_names) == 0:
            curr_label = None
            curr_data_loc = offset
            curr_label = dataset_names[curr_data_loc]
            curr_label = get_data_name_alias(curr_label)
            ylabels.append(curr_label)
        else:
            ylabels.append("")
    ax1.set_yticklabels([get_data_name_alias_for_plot_axes(lab) for lab in ylabels], rotation=90)
    ax1.yaxis.set_ticks_position('none')

    # Customize the grid: Add both major and minor grid lines
    ax1.grid(True, axis='x', which='major', linestyle=':', linewidth=0.75, color='grey')  # Major grid lines
    ax1.minorticks_on()  # Enable minor ticks
    ax1.grid(True, axis='x', which='minor', linestyle=':', linewidth=0.5, color='lightgray')  # Minor grid lines

    # Optional: Customize appearance
    ax1.invert_yaxis()  # Invert y-axis to display the first category at the top

    # Add legend
    ax1.legend()

    # Show plot
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    if x_domain_lim is not None:
        assert len(x_domain_lim) == 2
        ax1.set_xlim(x_domain_lim[0], x_domain_lim[1])#.0, 0.60)
    plt.tight_layout()
    
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass


def plot_confidence_interval_summary(save_path, center, lower_bnd, upper_bnd, center_label, title, criteria_name, domain_name):
    plt.figure(figsize=(15,5))
    plt.plot(center, marker='.', label=center_label)
    plt.plot(lower_bnd, marker='.', label="lower-bound")
    plt.plot(upper_bnd, marker='.', label="upper-bound")
    
    plt.title(title)
    plt.ylabel(criteria_name)
    plt.xlabel(domain_name)
    plt.legend()
    plt.grid()
    
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass
    

def make_scatter_and_stdErrOfMean_plot_overlay_vis(vals_by_label, save_path, title, x_label, y_label, alpha=0.5, nan_substitute=5., make_diff_plots=True):
    """
    References: 
    https://stackoverflow.com/questions/40837142/overlay-box-plots-on-bars
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
    """
    print("make_scatter_and_stdErrOfMean_plot_overlay_vis: vals_by_label == ", vals_by_label, flush=True)
    scatter_x_vals = []
    scatter_y_vals = []
    scatter_labels = []
    error_x = []
    error_y = []
    error_yerr = []
    error_labels = [""]
    for i, label in enumerate(vals_by_label.keys()):
        scatter_labels.append(label)
        scatter_y_vals.append(vals_by_label[label])
        perturbed_x_vals = [(2*i)+1.+(0.2*np.random.uniform()) for _ in vals_by_label[label]]
        scatter_x_vals.append(perturbed_x_vals)
        curr_mean = np.mean(vals_by_label[label])
        curr_se = np.std(vals_by_label[label])/np.sqrt(1.*len(vals_by_label[label]))
        error_labels.append(label+",\n m"+str(curr_mean)[:4]+"...,\n se"+str(curr_se)[:4]+"...")
        error_labels.append("") # add padding between labels to account for spacing between alg. plots
        error_x.append([(2*i)+1.])
        error_y.append([curr_mean])
        error_yerr.append([curr_se])
    
    ylim_max = np.max([[v if np.isfinite(v) else nan_substitute for v in vals_by_label[l]] for l in vals_by_label])
    ylim_max += ylim_max*0.5
    ylim_min = np.min([[v if np.isfinite(v) else nan_substitute for v in vals_by_label[l]] for l in vals_by_label])
    
    fig, ax = plt.subplots(1,1,figsize=(12,10))
    for (l,xs,ys) in zip(scatter_labels, scatter_x_vals, scatter_y_vals):
        ax.scatter(xs, ys, marker='.', alpha=alpha)
    ax.set_ylim([ylim_min, ylim_max])
    ax2 = ax.twinx()
    for (l,x,y,yE) in zip(error_labels, error_x, error_y, error_yerr): # see https://matplotlib.org/stable/gallery/statistics/errorbar_features.html
        ax2.errorbar(x, y, yerr=yE, capsize=4, alpha=alpha) # see https://www.geeksforgeeks.org/add-perpendicular-caps-to-error-bars-in-matplotlib/
    ax2.set_ylim(ax.get_ylim())
    ax.set_xticks(range(0,len(error_labels)), error_labels, rotation=70)
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    
    if make_diff_plots:
        for i, label1 in enumerate(vals_by_label.keys()):
            new_val_by_key_dict = dict()
            for j, label2 in enumerate(vals_by_label.keys()):
                if label1 != label2:
                    new_val_by_key_dict[label1+" - "+label2] = [v1 - v2 for (v1,v2) in zip(vals_by_label[label1], vals_by_label[label2])]
            save_folder = os.sep.join(save_path.split(os.sep)[:-1])
            save_file = save_path.split(os.sep)[-1]
            diff_save_path = save_folder + os.sep + label1+"_IMPROVEMENTS"
            if not os.path.exists(diff_save_path):
                os.mkdir(diff_save_path)
            diff_save_path = diff_save_path + os.sep + save_file
            make_scatter_and_stdErrOfMean_plot_overlay_vis(
                new_val_by_key_dict, 
                diff_save_path, 
                title+"\n vs "+label1+" Performance", 
                x_label, 
                y_label, 
                alpha=alpha, 
                nan_substitute=nan_substitute, 
                make_diff_plots=False
            )
    pass

def make_bar_and_whisker_plot_overlay_vis(vals_by_label, save_path, title, x_label, y_label, alpha=0.5, color="darkred"):
    """
    References: 
    https://stackoverflow.com/questions/40837142/overlay-box-plots-on-bars
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
    """
    ylim_max = np.max([vals_by_label[l] for l in vals_by_label])
    ylim_max += ylim_max*0.5
    
    fig, ax = plt.subplots()
    
    ax.bar(range(1, len(vals_by_label.keys())+1), [np.mean(vals_by_label[l]) for l in vals_by_label], align='center', alpha=alpha, color=color)
    ax.set_ylim([0, ylim_max])
    ax2 = ax.twinx()
    ax2.boxplot([vals_by_label[l] for l in vals_by_label])
    ax2.set_ylim(ax.get_ylim())
    ax.set_xticklabels([x for x in vals_by_label.keys()], rotation='vertical')
    
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass


def plot_scattered_results(x_vals, y_vals, save_path, title, x_label, y_label, x_eps=0., y_eps=0., alpha=0.5):
    # Plot data
    plt.figure(figsize=(10, 10))
    plt.scatter([x+random.gauss(0.,x_eps) for x in x_vals], [y+random.gauss(0.,y_eps) for y in y_vals], alpha=alpha)
    plt.title(title)
    plt.ylabel(y_label+" with eps="+str(y_eps))
    plt.xlabel(x_label+" with eps="+str(x_eps))
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass


def plot_training_loss(train_loss_list, save_path):
    # Plot data
    plt.figure(figsize=(8, 5))
    plt.plot(50 * np.arange(len(train_loss_list)), train_loss_list)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Training steps')
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass


def plot_heatmap(grid, save_path, title, ylabel, xlabel):
    fig, axarr = plt.subplots(1, 1, figsize=(10, 10))
    im1 = axarr.imshow(grid)
    axarr.set_title(title)
    axarr.set_ylabel(ylabel)
    axarr.set_xlabel(xlabel)
    axarr.set_xticks([])
    axarr.set_yticks([])
    plt.colorbar(im1, ax=axarr)
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass


def plot_reconstruction_comparisson(orig_feature_vals, pred_feature_vals, save_path):
    # Plot data
    plt.figure(figsize=(10, 10))
    plt.plot(orig_feature_vals, label="ground truth")
    plt.plot(pred_feature_vals, label="predicted")
    plt.title('Reconstructed Feature Comparisson')
    plt.ylabel('Feature Value')
    plt.xlabel('Feature')
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass
    

def plot_gc_est_comparisson(GC, GC_est, save_path, include_lags=False):
    num_est_driven_chans = None
    num_est_driving_chans = None
    num_true_driven_chans = None
    num_true_driving_chans = None
    num_true_lags = None
    num_est_lags = None
    if include_lags:
        if GC is not None:
            num_true_driven_chans = GC.shape[0]
            num_true_driving_chans = GC.shape[1]
            num_true_lags = GC.shape[2]
            GC = flatten_GC_estimate_with_lags(GC)
        if GC_est is not None:
            assert len(GC_est.shape) == 3
            num_est_driven_chans = GC_est.shape[0]
            num_est_driving_chans = GC_est.shape[1]
            num_est_lags = GC_est.shape[2]
            GC_est = flatten_GC_estimate_with_lags(GC_est)
        
    # Make figures
    fig, axarr = None, None
    if GC is None or GC_est is None:
        if include_lags:
            fig, axarr = plt.subplots(1, 1, figsize=(60, 5))
        else:
            fig, axarr = plt.subplots(1, 1, figsize=(10, 10))
    elif include_lags:
        fig, axarr = plt.subplots(2, 1, figsize=(60, 10))
    else: # not include_lags
        fig, axarr = plt.subplots(2, 1, figsize=(20, 10))

    if GC is not None:
        if GC_est is not None:
            im1 = axarr[0].imshow(GC)
            axarr[0].set_title('GC actual')
            axarr[0].set_ylabel('Affected series')
            axarr[0].set_xlabel('Causal series')
            axarr[0].set_xticks([])
            axarr[0].set_yticks([])
            plt.colorbar(im1, ax=axarr[0])
        else:
            im1 = axarr.imshow(GC)
            axarr.set_title('GC actual')
            axarr.set_ylabel('Affected series')
            axarr.set_xlabel('Causal series')
            axarr.set_xticks([])
            axarr.set_yticks([])
            plt.colorbar(im1, ax=axarr)
    else: 
        assert GC_est is not None
        im1 = axarr.imshow(GC_est)
        axarr.set_title('GC estimate')
        axarr.set_ylabel('Affected series')
        axarr.set_xlabel('Causal series')
        axarr.set_xticks([])
        axarr.set_yticks([])
        plt.colorbar(im1, ax=axarr)

    if GC is not None and GC_est is not None:
        im2 = axarr[1].imshow(GC_est)
        axarr[1].set_title('Estimated GC (fwd)')
        axarr[1].set_ylabel('Affected series')
        axarr[1].set_xlabel('Causal series')
        axarr[1].set_xticks([])
        axarr[1].set_yticks([])
        plt.colorbar(im2, ax=axarr[1])

    if include_lags: 
        if GC is not None:
            for l in range(1,num_true_lags):
                if GC_est is not None:
                    axarr[0].axvline(l*num_true_driving_chans-0.5, color='k')
                else:
                    axarr.axvline(l*num_true_driving_chans-0.5, color='k')
            if GC_est is not None:
                for l in range(1,num_est_lags):
                    if GC is not None:
                        axarr[1].axvline(l*num_est_driving_chans-0.5, color='k')
                    else:
                        axarr.axvline(l*num_est_driving_chans-0.5, color='k')
        else: 
            for l in range(1,num_est_lags):
                axarr.axvline(l*num_est_driving_chans-0.5, color='k')
    
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass


def plot_gc_est_comparissons_by_factor(GC, GC_est, save_path, include_lags=False):
    if GC is not None and GC_est is not None:
        for i, true_gc in enumerate(GC):
            for j, gc_est in enumerate(GC_est):
                plot_gc_est_comparisson(true_gc, gc_est, save_path[:-4]+"_factor"+str(j)+"_vs_trueGCFact"+str(i)+save_path[-4:], include_lags=include_lags)
    elif GC is None:
        assert GC_est is not None
        for j, gc_est in enumerate(GC_est):
            plot_gc_est_comparisson(None, gc_est, save_path[:-4]+"_factor"+str(j)+save_path[-4:], include_lags=include_lags) 
    else:
        assert GC_est is None   
        for j, gc in enumerate(GC):
            plot_gc_est_comparisson(gc, None, save_path[:-4]+"_factor"+str(j)+save_path[-4:], include_lags=include_lags) 
    pass


def plot_x_wavelet_comparisson(x, x_decomp_coeffs, x_approx, save_path):
    if torch.cuda.is_available():
        assert len(x.size()) == 1 # assumed to be size (temporal)
        assert len(x_decomp_coeffs[0].size()) == 1 # assumed to be size (num_coeffs, temporal)
        assert len(x_approx.size()) == 1 # assumed to be size (temporal)
        x = x.cpu().detach().numpy()
        x_decomp_coeffs = x_decomp_coeffs.cpu().detach().numpy()
        x_approx = x_approx.cpu().detach().numpy()
    else:
        assert len(x.shape) == 1 # assumed to be size (temporal)
        assert len(x_decomp_coeffs[0].shape) == 1 # assumed to be size (num_coeffs, temporal)
        assert len(x_approx.shape) == 1 # assumed to be size (temporal)

    # Make figures
    fig, axarr = plt.subplots(1+len(x_decomp_coeffs), 1, figsize=(15, 10))

    axarr[0].plot(x, label="true x")
    axarr[0].plot(x_approx, label="approx. x")
    axarr[0].set_title('True Signal vs Approximation')
    axarr[0].set_ylabel('Amplitude')
    axarr[0].set_xlabel('T')
    axarr[0].legend()

    for i, c in enumerate(x_decomp_coeffs):
        axarr[i+1].plot(c, label="level "+str(i))
        axarr[i+1].set_title("Wavelet Level "+str(i)+" Coefficients")
        axarr[i+1].set_ylabel('Amplitude')
        axarr[i+1].set_xlabel('T')
        axarr[i+1].legend()
    
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    
    # Make ZOOMED figures
    fig, axarr = plt.subplots(1+len(x_decomp_coeffs), 1, figsize=(15, 10))

    axarr[0].plot(x[:100], label="true x")
    axarr[0].plot(x_approx[:100], label="approx. x")
    axarr[0].set_title('True Signal vs Approximation (ZOOMED)')
    axarr[0].set_ylabel('Amplitude')
    axarr[0].set_xlabel('T')
    axarr[0].legend()

    for i, c in enumerate(x_decomp_coeffs):
        axarr[i+1].plot(c[:100], label="level "+str(i))
        axarr[i+1].set_title("Wavelet Level "+str(i)+" Coefficients (ZOOMED)")
        axarr[i+1].set_ylabel('Amplitude')
        axarr[i+1].set_xlabel('T')
        axarr[i+1].legend()

    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path[:-4]+"_ZOOMED.png")
    plt.close()
    pass


def plot_x_simulation_comparisson(x, x_sim, save_path):
    assert len(x.size()) == 3 # assumed to be size (batch size, temporal, channel)
    assert len(x_sim.size()) == 3 # assumed to be size (batch size, temporal, channel)
    num_channels = x.size()[2]

    # Make figures
    fig, axarr = plt.subplots(num_channels, 2, figsize=(5*num_channels, 5*num_channels))
    for i in range(num_channels):
        if x is not None:
            axarr[i,0].plot(x[0,:,i].cpu().detach().numpy())
            axarr[i,0].set_title('Actual Channel '+str(i))
            axarr[i,0].set_ylabel('Amplitude')
            axarr[i,0].set_xlabel('T')
        axarr[i,1].plot(x_sim[0,:,i].cpu().detach().numpy())
        axarr[i,1].set_title('Simulated Channel '+str(i))
        axarr[i,1].set_ylabel('Amplitude')
        axarr[i,1].set_xlabel('T')
    
    plt.tight_layout()
    plt.draw()
    plt.savefig(save_path)
    plt.close()
    pass


def plot_scatter(x, y, title, x_axis_name, y_axis_name, save_path): # <><><>
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, y)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_axis_name)
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_curve(output, title, x_axis_name, y_axis_name, save_path, domain_start=0): # <><><>
    fig1, ax1 = plt.subplots()
    ax1.plot([domain_start+i for i in range(len(output))], output)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_axis_name)
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_curve_comparisson(lists_of_curve_values, title, x_axis_name, y_axis_name, save_path, domain_start=0, label_root=""): # <><><>
    fig1, ax1 = plt.subplots()
    avg_curve = []
    for i, curve in enumerate(lists_of_curve_values):
        ax1.plot([domain_start+i for i in range(len(curve))], curve, label=label_root+str(i), alpha=0.5)
        if i == 0:
            avg_curve = avg_curve + curve
        else:
            avg_curve = [x1+x2 for (x1,x2) in zip(avg_curve, curve)]
    avg_curve = [x1/len(lists_of_curve_values) for x1 in avg_curve]
    ax1.plot([domain_start+i for i in range(len(avg_curve))], avg_curve, label="mean", alpha=0.5)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_axis_name)
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_curve_comparisson_from_dict(dict_of_curve_values, title, x_axis_name, y_axis_name, save_path, domain_start=0, label_root=""): # <><><>
    fig1, ax1 = plt.subplots()
    avg_curve = []
    for i, key in enumerate(dict_of_curve_values.keys()):
        curve = dict_of_curve_values[key]
        ax1.plot([domain_start+i for i in range(len(curve))], curve, label=label_root+key, alpha=0.5)
        if i == 0:
            avg_curve = avg_curve + curve
        else:
            avg_curve = [x1+x2 for (x1,x2) in zip(avg_curve, curve)]
    avg_curve = [x1/len(list(dict_of_curve_values.keys())) for x1 in avg_curve]
    ax1.plot([domain_start+i for i in range(len(avg_curve))], avg_curve, label="mean", alpha=0.5)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_axis_name)
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass

def plot_all_signal_channels(x, title, save_path, x_axis_name="T", y_axis_name="Channel Value", zoom=None): # <><><>
    fig1, ax1 = plt.subplots(figsize=(15, 10))
    ax1.plot(x.T)
    ax1.set_xlabel(x_axis_name)
    ax1.set_ylabel(y_axis_name)
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path+os.sep+title+".png")
    plt.close()

    if zoom is not None:    
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        ax2.plot(x[:,:zoom].T)
        ax2.set_xlabel(x_axis_name)
        ax2.set_ylabel(y_axis_name)
        ax2.set_title(title)
        plt.legend()
        plt.draw()
        fig2.savefig(save_path+os.sep+title+"_ZOOMED.png")
        plt.close()

        fig3, ax3 = plt.subplots(figsize=(15, 10))
        ax3.plot(x[:,:2*zoom].T)
        ax3.set_xlabel(x_axis_name)
        ax3.set_ylabel(y_axis_name)
        ax3.set_title(title)
        plt.legend()
        plt.draw()
        fig3.savefig(save_path+os.sep+title+"_partiallyZOOMED.png")
        plt.close()
    pass


def plot_system_state_score_comparisson(save_path, scores, title, colors, markers, labels):
    num_states = scores.shape[0]
    total_recording_len = scores.shape[1]
    state_recording_len = total_recording_len // num_states

    fig1, ax1 = plt.subplots()
    for state_id in range(num_states):
        ax1.plot(scores[state_id,:], color=colors[state_id], marker=markers[state_id], label=labels[state_id], alpha=0.5)
        if state_id > 0:
            plt.axvline(x=state_id*state_recording_len, color='k', linestyle='dashed')
    ax1.set_xlabel("Recording Time ID")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass


def plot_avg_system_state_score_comparisson(save_path, scores, true_label_traces, title, colors, markers, labels):
    avg_scores_at_each_time_step = np.zeros(scores[0].shape)
    for score_recording in scores:
        avg_scores_at_each_time_step = avg_scores_at_each_time_step + score_recording
    avg_scores_at_each_time_step = avg_scores_at_each_time_step / (1.*len(scores))
    
    avg_true_label_trace = np.zeros(true_label_traces[0].shape)
    for label_trace in true_label_traces:
        avg_true_label_trace = avg_true_label_trace + label_trace
    avg_true_label_trace = avg_true_label_trace / (1.*len(true_label_traces))
    
    num_states = avg_scores_at_each_time_step.shape[0]
    total_recording_len = avg_scores_at_each_time_step.shape[1]

    fig1, ax1 = plt.subplots(1,1,figsize=(15,10))
    for score_recording in scores:
        for state_id in range(num_states):
            ax1.plot(score_recording[state_id,:], color=colors[state_id], marker=markers[state_id], alpha=0.025) # plot specific score recordings as very light (unlabeled) lines in the background
            
    for state_id in range(num_states):
        ax1.plot(avg_scores_at_each_time_step[state_id,:], color=colors[state_id], marker=markers[state_id], label="avg_pred_"+labels[state_id], alpha=0.5)
        ax1.plot(avg_true_label_trace[state_id,:], color=colors[state_id], marker=markers[state_id], label="true_"+labels[state_id], alpha=0.5, linestyle='dotted')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title)
    ax1.set_ylim(-1,2.5)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass


def plot_estimated_vs_true_curve(save_path, est_curve, true_curve, title, x_label, y_label):
    fig1, ax1 = plt.subplots()
    ax1.plot(true_curve, color='k', marker='+', label='true', alpha=0.5)
    ax1.plot(est_curve, color='salmon', marker='x', label='estimated', alpha=0.5)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    plt.legend()
    plt.draw()
    fig1.savefig(save_path)
    plt.close()
    pass