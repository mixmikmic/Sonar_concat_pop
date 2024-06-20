# # Odometry Reconstruction Accuracy Results
# 
#  * **Warning:** quite a bit of copypasta from the 'DepthAnalysis' notebook.
# 

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

get_ipython().magic('matplotlib inline')

from matplotlib import rc

# Enables full LaTeX support in the plot text.
# Requires a full-fledged LaTeX installation on your system, accessible via PATH.
rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (16, 5)
matplotlib.rcParams.update({'font.size': 20})


def gen_plots(root, part, out_dir, **kw):
    file_pattern = 'k-99999-kitti-odometry-{sequence_id:02d}-offset-0-depth-precomputed-{depth}-'                    'voxelsize-0.0500-max-depth-m-20.00-dynamic-mode-NO-direct-ref-'                    'with-fusion-weights-{part}.csv'
    base = os.path.join(root, file_pattern)
    save_to_disk = kw.get('save_to_disk', True)
    
    sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    metrics = ['input', 'fusion']
    depths = ['elas', 'dispnet']
    
    # Load the data and prepare some positioning and color info
    # (Since we're doing a nontrivial plot, which means we need to do a lot of
    #  stuff manually.)
    box_positions = []
    box_colors = []
    columns = []
    box_offset = 0.0
    INNER_GAP = 0.75
    SEQUENCE_GAP = 1.0
    GROUP_SIZE = len(depths) * len(metrics)
    
    colors = {
        'elas': {
            'input': 'C0',
            'fusion': 'C1',
        },
        'dispnet': {
            'input': 'C2',
            'fusion': 'C3'
        }
    }
    
    def setup_xaxis_legend(ax, **kw):
        bp_np = np.array(box_positions)
        alt_ticks = bp_np[np.arange(len(bp_np)) % GROUP_SIZE == 0] + (INNER_GAP*(GROUP_SIZE-1.0)/2.0)
        ax.set_xticks(alt_ticks)
        ax.set_xticklabels("{:02d}".format(sid) for sid in sequences)
        ax.set_xlabel("Sequence")

        ax.set_ylim([0.0, 1.0])

        for patch, color in zip(boxplot['medians'], box_colors):
            patch.set_color(color)    

        for patch, color in zip(boxplot['boxes'], box_colors):
            patch.set_color(color)

        # Ugly, but required since every box has two whiskers and two caps...
        for idx, (whisker, cap) in enumerate(zip(boxplot['whiskers'], boxplot['caps'])):
            cap.set_color(box_colors[idx%(2*GROUP_SIZE) // 2])
            whisker.set_color(box_colors[idx%(2*GROUP_SIZE) // 2])   

        # Dummies for showing the appropriate legend
        ax.plot([0.0], [-1000], label="ELAS input", color=colors['elas']['input'])
        ax.plot([0.0], [-1000], label="ELAS fused", color=colors['elas']['fusion'])
        ax.plot([0.0], [-1000], label="DispNet input", color=colors['dispnet']['input'])
        ax.plot([0.0], [-1000], label="DispNet fused", color=colors['dispnet']['fusion'])
        ax.legend(loc=kw.get('legendloc', 'lower left'))

        ax.grid('off')
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)
        
    def save_fig(f, fname):
        print("Saving figure to [{}]... ".format(fname), end='')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        f.savefig(os.path.join(out_dir, fname + '.png'), bbox_inches='tight')
        f.savefig(os.path.join(out_dir, fname + '.eps'), bbox_inches='tight')
        print("\rSaved figure to [{}].    ".format(fname))
        
    def compute_metrics(dataframe, metric):               
        # Do not count frames with no pixels in them. This would distort the 
        # dynamic reconstruction metrics due to frames containing no objects.
        ok = (dataframe['{}-total-3.00-kitti'.format(metric)] != 0)

        err = dataframe['{}-error-3.00-kitti'.format(metric)][ok]
        tot = dataframe['{}-total-3.00-kitti'.format(metric)][ok]
        mis = dataframe['{}-missing-3.00-kitti'.format(metric)][ok]
        cor = dataframe['{}-correct-3.00-kitti'.format(metric)][ok]
        mis_sep = dataframe['{}-missing-separate-3.00-kitti'.format(metric)][ok]

        acc_perc = cor / (tot - mis)
        completeness = 1.0 - (mis_sep / tot)
        
        return acc_perc, completeness
    
    def setup_agg_plot(ax, boxplot):
        # Aesthetic crap
        ax.set_ylim([0.4, 1.01])
        plt.minorticks_on()
        plt.xticks(rotation=45, ha='right')

        for patch in boxplot['medians']:
            patch.set_color('black')
        for patch in boxplot['boxes']:
            patch.set_color('black')
        for patch in boxplot['whiskers']:
            patch.set_color('black')

        ax.set_xticklabels(["ELAS input", "ELAS fused", "DispNet input", "DispNet fused"])
        ax.grid('off')
        ax.yaxis.grid(True, linestyle='-', which='major', color='gray', alpha=0.75)
        ax.yaxis.grid(True, linestyle='-', which='minor', color='lightgrey', alpha=0.75)
    
    res = {}
    res_completeness = {}
    
    # Aggregated for all the sequences.
    res_acc_agg = {}
    res_completeness_agg = {}
    
    for sequence_id in sequences:
        for depth in depths:
            # Part dictates what we are evaluating: dynamic or static parts
            fname = base.format(sequence_id=sequence_id, depth=depth, part=part)
            df = pd.read_csv(fname)
#             print("{} frames in sequence #{}-{}".format(len(df), sequence_id, depth))
            
            for metric in metrics:
                key = "{}-{}-{:02d}".format(metric, depth, sequence_id)
                agg_key = "{}-{}".format(metric, depth)
                if not agg_key in res_acc_agg:
                    res_acc_agg[agg_key] = []
                    res_completeness_agg[agg_key] = []
                
                acc_perc, completeness = compute_metrics(df, metric)
                res[key] = acc_perc
                res_completeness[key] = completeness
                res_acc_agg[agg_key] = res_acc_agg[agg_key] + acc_perc.tolist()
                res_completeness_agg[agg_key] = res_completeness_agg[agg_key] + completeness.tolist()
                
                box_colors.append(colors[depth][metric])
                
                columns.append(key)
                box_positions.append(box_offset)
                box_offset += INNER_GAP
            
        box_offset += SEQUENCE_GAP
                
#     res_acc_all = [entry for (key, sublist) in res.items() for entry in sublist]
        
    print("Data read & aggregated OK.")
    
    print("Agg meta-stats:")
    for k, v in res_acc_agg.items():
        print(k, len(v))
    
    
    ################################################################################
    # Accuracy plots
    ################################################################################
    res_df = pd.DataFrame(res)    
    FIG_SIZE = (16, 6)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    (ax, boxplot) = res_df.boxplot(columns, showfliers=False, 
                                   return_type='both',
                                   widths=0.50, 
                                   ax=ax, 
#                                    patch_artist=True,  # Enable fill
                                   positions=box_positions)
    setup_xaxis_legend(ax)
    ax.set_ylabel("Accuracy", labelpad=15)
    ax.set_ylim([0.3, 1.01])
    if save_to_disk:
        save_fig(fig, 'odo-acc-{}'.format(part))
    
    ################################################################################
    # Aggregate accuracy plots
    ################################################################################
    res_acc_agg_df = pd.DataFrame(res_acc_agg)
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(1, 1, 1)
    agg_cols = ['input-elas', 'fusion-elas', 'input-dispnet', 'fusion-dispnet']
    (ax, boxplot) = res_acc_agg_df.boxplot(agg_cols, showfliers=False,
                                           return_type='both',
                                           widths=0.25,
                                           ax=ax)
    ax.set_ylabel("Accuracy", labelpad=15)
    setup_agg_plot(ax, boxplot)

    print("Textual results: ")
    for col in agg_cols:
        print(col, ":", res_acc_agg_df[col].mean())
    
    if save_to_disk:
        save_fig(fig, 'odo-acc-agg-{}'.format(part))
    
    ################################################################################
    # Completeness plots
    ################################################################################
    res_completeness_df = pd.DataFrame(res_completeness)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    
    (ax, boxplot) = res_completeness_df.boxplot(columns, showfliers=False, 
                                               return_type='both',
                                               widths=0.50, 
                                               ax=ax, 
            #                                    patch_artist=True,  # Enable fill
                                               positions=box_positions)
    
    setup_xaxis_legend(ax)
    ax.set_ylim([0.3, 1.01])
    ax.set_ylabel("Completeness")
    
    if save_to_disk:
        save_fig(fig, 'odo-completeness-{}'.format(part))
        
    ################################################################################
    # Aggregate completeness plots
    ################################################################################
    res_completeness_agg_df = pd.DataFrame(res_completeness_agg)
    fig = plt.figure(figsize=(4,6))
    ax = fig.add_subplot(1, 1, 1)
    agg_cols = ['input-elas', 'fusion-elas', 'input-dispnet', 'fusion-dispnet']
    (ax, boxplot) = res_completeness_agg_df.boxplot(agg_cols, showfliers=False,
                                                    return_type='both',
                                                    widths=0.25,
                                                    ax=ax)
    ax.set_ylabel("Completeness", labelpad=15)
    setup_agg_plot(ax, boxplot)

    print("Textual results: ")
    for col in agg_cols:
        print(col, ":", res_completeness_agg_df[col].mean())
    
    if save_to_disk:
        save_fig(fig, 'odo-completeness-agg-{}'.format(part))
    
                
        
save = True
out_dir = '../fig'
gen_plots('../csv/odo-res', 'static-depth-result', out_dir, save_to_disk=save)
gen_plots('../csv/odo-res', 'dynamic-depth-result', out_dir, save_to_disk=save)








# # Reduced Framerate Results
# 
# Copypasta-rich as well.
# 

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

get_ipython().magic('matplotlib inline')

from matplotlib import rc
# Enable full LaTeX support in plot text. Requires a full-fledged LaTeX installation
# on your system, accessible via PATH.
rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (16, 6)
matplotlib.rcParams.update({'font.size': 16})


out_dir = '../fig'

def gen_plots(root, part):
    file_pattern = 'k-99999-kitti-odometry-{sequence_id:02d}-offset-0-depth-precomputed-{depth}-'                    'voxelsize-0.0500-max-depth-m-20.00-dynamic-mode-NO-direct-ref-'                    'with-fusion-weights{fuse_every}-{part}.csv'
    base = os.path.join(root, file_pattern)
    res = {}
    res_completeness = {}
    
    sequence_id = 9
    fuse_every_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    metrics = ['input', 'fusion']
    # TODO eval on both
#     depths = ['elas', 'dispnet']
    depths = ['dispnet']
    
    # Load the data and prepare some positioning and color info
    # (Since we're doing a nontrivial plot, which means we need to do a lot of
    #  stuff manually.)
    box_positions = []
    box_colors = []
    columns = []
    box_offset = 0.0
    INNER_GAP = 0.75
    SEQUENCE_GAP = 1.0
    GROUP_SIZE = len(depths) * len(metrics)
    
    colors = {
        'elas': {
            'input': 'C2',
            'fusion': 'C3',
        },
        'dispnet': {
            'input': 'C0',
            'fusion': 'C1'
        }
    }
    
    def setup_xaxis_legend(ax, **kw):
        bp_np = np.array(box_positions)
        alt_ticks = bp_np[np.arange(len(bp_np)) % GROUP_SIZE == 0] + (INNER_GAP*(GROUP_SIZE-1.0)/2.0)
        ax.set_xticks(alt_ticks)
        ax.set_xticklabels("{:02d}".format(k) for k in fuse_every_vals)
        ax.set_xlabel("$k$ (Fusion every $k$th frame)")

        ax.set_ylim([0.0, 1.0])

        for patch, color in zip(boxplot['medians'], box_colors):
            patch.set_color(color)    

        for patch, color in zip(boxplot['boxes'], box_colors):
            patch.set_color(color)

        # Ugly, but required since every box has two whiskers and two caps...
        for idx, (whisker, cap) in enumerate(zip(boxplot['whiskers'], boxplot['caps'])):
            cap.set_color(box_colors[idx%(2*GROUP_SIZE) // 2])
            whisker.set_color(box_colors[idx%(2*GROUP_SIZE) // 2])   

        # Dummies for showing the appropriate legend
#         ax.plot([0.0], [-1000], label="ELAS input", color=colors['elas']['input'])
#         ax.plot([0.0], [-1000], label="ELAS fused", color=colors['elas']['fusion'])
        ax.plot([0.0], [-1000], label="DispNet input", color=colors['dispnet']['input'])
        ax.plot([0.0], [-1000], label="DispNet fused", color=colors['dispnet']['fusion'])
        ax.legend(loc=kw.get('legendloc', 'lower left'))

        ax.grid('off')
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)
        
    def save_fig(f, fname):
        print("Saving figure to [{}]... ".format(fname), end='')
        f.savefig(os.path.join(out_dir, fname + '.png'), bbox_inches='tight')
        f.savefig(os.path.join(out_dir, fname + '.eps'), bbox_inches='tight')
        print("\rSaved figure to [{}] in [{}].    ".format(fname, out_dir))
        
    
    for fuse_every in fuse_every_vals:
        for depth in depths:
            # Part dictates what we are evaluating: dynamic or static parts
            fuse_every_val = "" if fuse_every == 1 else "-fuse-every-{}".format(fuse_every)
            fname = base.format(sequence_id=sequence_id, depth=depth, fuse_every=fuse_every_val, part=part)
            df = pd.read_csv(fname)
#             print("DF OK", fuse_every, depth, len(df))
            
            for metric in metrics:
                key = "{}-{}-{:02d}".format(metric, depth, fuse_every)
                
                # Do not count frames with no pixels in them. This would distort the 
                # dynamic reconstruction metrics due to frames containing no objects.
                ok = (df['{}-total-3.00-kitti'.format(metric)] != 0)

                err = df['{}-error-3.00-kitti'.format(metric)][ok]
                tot = df['{}-total-3.00-kitti'.format(metric)][ok]
                mis = df['{}-missing-3.00-kitti'.format(metric)][ok]
                cor = df['{}-correct-3.00-kitti'.format(metric)][ok]
                mis_sep = df['{}-missing-separate-3.00-kitti'.format(metric)][ok]
                
                err = err[50:1050]
                tot = tot[50:1050]
                mis = mis[50:1050]
                cor = cor[50:1050]
                mis_sep = mis_sep[50:1050]
                

                acc_perc = cor / (tot - mis)
                res[key] = acc_perc
                
#                 print(fuse_every, depth, acc_perc.mean())
                
                completeness = 1.0 - (mis_sep / tot)
                res_completeness[key] = completeness
                
                box_colors.append(colors[depth][metric])
                
                columns.append(key)
                box_positions.append(box_offset)
                box_offset += INNER_GAP
            
        box_offset += SEQUENCE_GAP
        
    ################################################################################
    # Accuracy plots
    ################################################################################
    res_df = pd.DataFrame(res)    
    FIG_SIZE = (16, 6)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    (ax, boxplot) = res_df.boxplot(columns, showfliers=False, 
                                   return_type='both',
                                   widths=0.50, 
                                   ax=ax, 
#                                    patch_artist=True,  # Enable fill
                                   positions=box_positions)
    setup_xaxis_legend(ax)
    ax.set_ylabel("Accuracy", labelpad=15)
    ax.set_ylim([0.3, 1.01])
    save_fig(fig, 'low-time-res-acc-{}'.format(part))
    
    ################################################################################
    # Completeness plots
    ################################################################################
    res_completeness_df = pd.DataFrame(res_completeness)
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(1, 1, 1)
    
    (ax, boxplot) = res_completeness_df.boxplot(columns, showfliers=False, 
                                               return_type='both',
                                               widths=0.50, 
                                               ax=ax, 
            #                                    patch_artist=True,  # Enable fill
                                               positions=box_positions)
    
    setup_xaxis_legend(ax)
    ax.set_ylim([0.3, 1.01])
    ax.set_ylabel("Completeness")
    save_fig(fig, 'low-time-res-com-{}'.format(part))
                
        
gen_plots('../csv/low-time-res-res', 'static-depth-result')





# # Pretty Depth Maps
# 

import numpy as np
import re
import sys

from scipy import misc

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


DIR = "/home/barsana/datasets/kitti/odometry-dataset/sequences/05/precomputed-depth-dispnet-png/"


for i in range(1, 100):
    fpath = "{}{:06d}.png".format(DIR, i)
    fpath_pretty = "{}{:06d}-viridis.png".format(DIR, i)
    img = plt.imread(fpath)
    
    plt.imsave(fpath_pretty, img, cmap='viridis')
    



D_ELAS = "/home/barsana/datasets/kitti/odometry-dataset/sequences/06/image_2/"

elas = D_ELAS + "000028_disp.pgm"
elas_dispmap = plt.imread(elas)
plt.imshow(elas_dispmap, cmap='viridis')
plt.imsave(D_ELAS + "0000028_disp-viridis.png", elas_dispmap, cmap='viridis')





# # Voxel GC Stats
# 
# Quantitatively evaluating the memory consumption and accuracy of our reconstructions post voxel decay.
# 
# Goals:
#  * Show memory usage goes down with the threshold k
#  * Show accuracy is OK, and maybe only goes down a little
#  * Show that completeness only goes down a little with super k-s
#  * It's probably best to have side-by-side plots of DispNet and ELAS
# 

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

get_ipython().magic('matplotlib inline')

from matplotlib import rc

# Enable full LaTeX support in plot text. Requires a full-fledged LaTeX installation
# on your system, accessible via PATH.
rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (16, 6)
matplotlib.rcParams.update({'font.size': 20})


root = os.path.join('..', 'csv', 'decay-experiments')


def stitch(base_tpl):
    """Used to stitch together data from multiple CSV dumps.
    
    Useful if you want X frames but the system crashes from a stupid bug
    when you're 99% done.
    """
    components = ['dynamic-depth-result', 'static-depth-result', 'memory']
    bits = ['-A', '-B']
    offsets = [0, 864]
    for c in components:
        curr_frame_off = 0
        curr_memory_off = 0  
        # No need for saved memory tracking since so far we're just stitching sequences with no decay.
        
        cum_df = None
        print("\n\nStarting component: {}".format(c))
        
        for bit, offset in zip(bits, offsets):
            fk = 'frame_id' if c == 'memory' else 'frame'
            fpath = os.path.join(root, base_tpl.format(bit, offset, c))
            df = pd.read_csv(fpath)
            
            print(len(df))
            print("Last:", df[fk].iloc[-1])
            
            print("Adding {} to frame ids by key {}...".format(curr_frame_off, fk))
            df[fk] += curr_frame_off

            if c == 'memory':
                df['memory_usage_bytes'] += curr_memory_off
            
            if cum_df is None:
                cum_df = df
            else:
                cum_df = pd.concat([cum_df, df])
                
            print(c, "Len:", len(cum_df))
           
            curr_frame_off += df[fk].iloc[-1] 
            if c == 'memory':
                curr_memory_off = df['memory_usage_bytes'].iloc[-1]
                print("Mem offset: ", curr_memory_off)
        
        out_fname = base_tpl.format("", 0, c)
        
        out_fpath = os.path.join(root, out_fname)
        cum_df.index = cum_df[fk]
        del cum_df[fk]
        
        cum_df.to_csv(out_fpath)

# stitch("k-0{}-kitti-odometry-09-offset-{}-depth-precomputed-dispnet-voxelsize-0.0500-max-depth-m-20.00-dynamic-mode-NO-direct-ref-NO-fusion-weights-{}.csv")


acc_fname_template = 'k-{}-kitti-odometry-09-offset-0-depth-precomputed-{}-'                       'voxelsize-0.0500-max-depth-m-20.00-dynamic-mode-NO-direct-ref-'                      'NO-fusion-weights-static-depth-result.csv'
mem_fname_template = 'k-{}-kitti-odometry-09-offset-0-depth-precomputed-{}-'                       'voxelsize-0.0500-max-depth-m-20.00-dynamic-mode-NO-direct-ref-'                      'NO-fusion-weights-memory.csv'        

ks = [0, 1, 2, 3, 5, 8, 10]
depths = ['dispnet', 'elas']

memory = {}
frame_lim = 1000

stats = {
}

for depth in depths:
    memory[depth] = {}
    stats[depth] = {
        'k': [],
        'accuracy': [],
        'completeness': [],
        'f1': [],
        'mem-gb': [],
    }
    
    for k in ks:
        acc_fname = acc_fname_template.format(k, depth)
        mem_fname = mem_fname_template.format(k, depth)
        acc_fpath = os.path.join(root, acc_fname)
        mem_fpath = os.path.join(root, mem_fname)
        BYTE_TO_GB = 1.0 / (1024 * 1024 * 1024)
        
        df_acc = pd.read_csv(acc_fpath)
        df_mem = pd.read_csv(mem_fpath)
        mem_raw = df_mem['memory_usage_bytes'][:frame_lim]
        memory[depth]['$k_\\textrm{{weight}} = {}$'.format(k)] = mem_raw * BYTE_TO_GB

        total_gt = df_acc['fusion-total-3.00-kitti']
        err = df_acc['fusion-error-3.00-kitti'] / (total_gt - df_acc['fusion-missing-3.00-kitti'])
        completeness = (1.0 - df_acc['fusion-missing-separate-3.00-kitti'] / total_gt)
        
        err_m = err.mean()
        acc_m = (1.0 - err).mean()
        com_m = completeness.mean()
        # Not super rigorous, but does combine the two metrics somewhat meaningfully...
        poor_man_f1 = 2 * (acc_m * com_m) / (acc_m + com_m)
        
        stats[depth]['k'].append(k)
        stats[depth]['accuracy'].append(acc_m)
        stats[depth]['completeness'].append(com_m)
        stats[depth]['f1'].append(poor_man_f1)
        stats[depth]['mem-gb'].append(mem_raw[mem_raw.index[-1]] * BYTE_TO_GB)
       


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
ylim = [0, 4.0]

# This is dirty but does the job
ordered_cols = ['$k_\\textrm{{weight}} = {}$'.format(i) for i in ks]

df_disp = pd.DataFrame(memory['dispnet'])
df_disp.plot(y=ordered_cols, ax=ax1, legend=False)
ax1.set_ylim(ylim)
ax1.set_xlabel("Frame")
ax1.set_ylabel("Memory usage (GiB)")
ax1.set_title("DispNet depth maps")
# ax1.legend(loc='upper left')
ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

df_elas = pd.DataFrame(memory['elas'])
df_elas.plot(y=ordered_cols, ax=ax2, legend=False)
ax2.set_ylim(ylim)
ax2.set_xlabel("Frame")
# ax2.set_ylabel("Memory usage (GiB)")
ax2.set_title("ELAS depth maps")
# ax2.legend(loc='upper left')
ax2.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)

for i in ks:
    key = '$k_\\textrm{{weight}} = {}$'.format(i)
    key_short = '$k_\\textrm{{w}} = {}$'.format(i)
    ax1.text(df_disp.index[-1] + 25, df_disp[key].iloc[-1] - 0.05, key_short)
    ax2.text(df_elas.index[-1] + 25, df_elas[key].iloc[-1] - 0.05, key_short)

plt.subplots_adjust(right=1.6)
# fig.suptitle("Memory usage under varying voxel GC thresholds")
# plt.tight_layout()

fig.savefig('../fig/recon-over-time.eps', bbox_inches='tight')
fig.savefig('../fig/recon-over-time.png', bbox_inches='tight')


mem_col = 'C3'

# Quad-plot = default, in thesis.
# Non-quad, i.e., line => saves a tiny bit of space for the paper
quad_plot = False

if quad_plot:
    ROWS = 2
    COLS = 2
else:
    ROWS = 1
    COLS = 4

y_names = {
    'accuracy': 'Accuracy',
    'completeness': 'Completeness',
    'f1': 'F1-Score',
    'mem-gb': 'Memory Usage (GiB)'
}

MAX_GB = 4.5

def mk_acc_mem_plot(ax, stats, depth, key, label):
    ax.plot(stats[depth]['k'], stats[depth][key], label=label)
    ax_mem = ax.twinx()
    mem = np.array(stats[depth]['mem-gb'])
    
    ax_mem.bar(stats[depth]['k'], mem, label="Memory Usage", 
               color=mem_col, fill=True, alpha=0.5)

    ax_mem.set_ylabel('Memory Usage (GiB)', color=mem_col, labelpad=15)
    ax_mem.set_ylim([0, MAX_GB])
    for i, v in enumerate(mem):
        ax_mem.text(stats[depth]['k'][i] - 0.65, v + 0.05, "{:.1f} GiB".format(v),
                    color=mem_col, fontdict={'size': 12})
        
    ax.set_ylim([0.4, 1.0])
    ax.set_xlim([-1, 11])
    ax.set_xticks(np.arange(0, 11))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)
    
#     handles1, labels1 = ax.get_legend_handles_labels()
#     l = plt.legend(handles1, labels1, loc='lower left')
#     ax_mem.add_artist(l)  # Make the legend stay on top
    
    ax.set_ylabel(y_names[key], labelpad=10, color='C0')
    ax.set_xlabel("$k_\\textrm{weight}$")
    ax.set_title(label)

keys = ['accuracy', 'completeness', 'f1', 'mem-gb']
# for key in keys:#, 'mem-gb']):
#     fig, (ax_d, ax_e) = plt.subplots(1, 2, figsize=(12, 4))
    
#     mk_acc_mem_plot(ax_d, stats, 'dispnet', key, "DispNet")
#     mk_acc_mem_plot(ax_e, stats, 'elas', key, "ELAS")
    
fig, axes = plt.subplots(ROWS, COLS, figsize=(6 * COLS, 6 * ROWS))
for ax, key in zip(np.ravel(axes), keys):
    ax.plot(stats['elas']['k'], stats['elas'][key], '-x', label='ELAS')
    ax.plot(stats['dispnet']['k'], stats['dispnet'][key], '-^', label='DispNet')
    
    ax.set_xlabel("$k_\\textrm{weight}$")
    ax.set_ylabel(y_names[key], labelpad=10)
    
    if key != 'mem-gb':
        ax.set_ylim([0.4, 1.0])
    else:
        ax.set_ylim(0, MAX_GB)
        
    ax.set_xlim([-1, 11])
    ax.set_xticks(np.arange(0, 11))
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.75)
    ax.legend(loc='lower left')

    
#     left  = 0.125  # the left side of the subplots of the figure
#     right = 0.9    # the right side of the subplots of the figure
#     bottom = 0.1   # the bottom of the subplots of the figure
#     top = 0.9      # the top of the subplots of the figure
#     wspace = 0.2   # the amount of width reserved for blank space between subplots,
#                    # expressed as a fraction of the average axis width
#     hspace = 0.2   # the amount of height reserved for white space between subplots,
#                    # expressed as a fraction of the average axis height
#     fig.subplots_adjust(hspace = 0.5)

fig.tight_layout()
name = 'recon-acc-quad'#.format(key)
fig.savefig('../fig/{}.eps'.format(name), bbox_inches='tight')
fig.savefig('../fig/{}.png'.format(name), bbox_inches='tight')





# # Analyzing KITTI-tracking data and putting it in a table
# 
# (And maybe boxplots if sensible.)
# 
# Warning: copypasta from the 'StaticDepthAnalysis' notebook.
# 

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

get_ipython().magic('matplotlib inline')

from matplotlib import rc
# Enable full LaTeX support in plot text. Requires a full-fledged LaTeX installation
# on your system, accessible via PATH.
rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (16, 6)
matplotlib.rcParams.update({'font.size': 16})


out_dir = '../fig'

def gen_plots(root, part, eval_completeness):
    # if 'eval_completeness' is false, eval accuracy.
    # TODO maybe also compute results for dynamic parts.
    file_pattern = 'k-99999-kitti-tracking-sequence-{sequence_id:04d}--offset-0-'                       'depth-precomputed-{depth}-voxelsize-0.0500-max-depth-m-20.00-'                     '{fusion}-NO-direct-ref-with-fusion-weights-{part}.csv'
    base = os.path.join(root, file_pattern)
    res = {}
    res_completeness = {}
    
    sequences = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fusions = ['NO-dynamic', 'dynamic-mode']
    metrics = ['input', 'fusion']
    depths = ['dispnet', 'elas']
    
    res = {}
    res_completeness = {}
    
    # Not a proper header. The real one is in the thesis tex, since it's nested
    # and pointless to generate from code.
    for depth in depths:
        for fusion in fusions:
            for metric in metrics:
                dyn_str = 'Dyn' if fusion == 'dynamic-mode' else 'No-dyn'
                if metric == 'input' and fusion == 'NO-dynamic':
                    print('Input-{} {} & '.format(depth, dyn_str), end='')
                    
                if metric == 'fusion':
                    print('Fusion-{} {} &'.format(depth, dyn_str), end='')
    print()
    
    acc_perc_agg = {}
    completeness_agg = {}
    
    # Yowza, that's a lot of loop nesting!
    for sequence_id in sequences:
        seq_count = -1
        
        print('{:02d} &'.format(sequence_id), end='')
        for depth in depths:
            best_key = None
            best_score = -1.0
            
            for fusion in fusions:
                fname = base.format(sequence_id=sequence_id, depth=depth, 
                                        fusion=fusion, part=part)
                df = pd.read_csv(fname)
            
                for metric in metrics:
                    key = "{}-{}-{}-{:02d}".format(metric, depth, fusion, sequence_id)
                    cross_seq_key = "{}-{}-{}".format(metric, depth, fusion)

                    # Do not count frames with no pixels in them. This would distort the 
                    # dynamic reconstruction metrics due to frames containing no objects.
                    ok = (df['{}-total-3.00-kitti'.format(metric)] != 0)

                    err = df['{}-error-3.00-kitti'.format(metric)][ok]
                    tot = df['{}-total-3.00-kitti'.format(metric)][ok]
                    mis = df['{}-missing-3.00-kitti'.format(metric)][ok]
                    cor = df['{}-correct-3.00-kitti'.format(metric)][ok]
                    mis_sep = df['{}-missing-separate-3.00-kitti'.format(metric)][ok]

                    acc_perc = cor / (tot - mis)
                    # When evaluating dynamic parts, sometimes we encounter cases with
                    # e.g., very distant cars where tot == mis.
                    acc_perc = acc_perc[~np.isnan(acc_perc)]
                    completeness = 1.0 - (mis_sep / tot)
                    
                    if cross_seq_key not in acc_perc:
                        acc_perc_agg[cross_seq_key] = []
                        completeness_agg[cross_seq_key] = []
                        
                    acc_perc_agg[cross_seq_key] += acc_perc.tolist()
                    completeness_agg[cross_seq_key] += completeness.tolist()
                    
                    if eval_completeness:
                        res[key] = completeness
                    else:
                        res[key] = acc_perc
                    
                    
                    mean_acc_perc = acc_perc.mean()
                    mean_com_perc = completeness.mean()
                    
                    # The input should be the same in dynamic and non-dynamic mode.
                    if not (metric == 'input' and fusion == 'dynamic-mode'):
                        if eval_completeness:
                            # Compute and display completeness
                            if mean_com_perc > best_score:
                                best_score = mean_com_perc
                                best_key = key
                        else:
                            # Compute and display accuracy
                            if mean_acc_perc > best_score:
                                best_score = mean_acc_perc
                                best_key = key
                    
                    if -1 == seq_count:
                        seq_count = len(df)
                    elif seq_count != len(df):
                        print("Warning: inconsistent lengths for sequence {:04d}".format(sequence_id))
                        print(sequence_id, depth, fusion, metric, len(df))
                  
            for fusion in fusions:
                for metric in metrics:
                    key = "{}-{}-{}-{:02d}".format(metric, depth, fusion, sequence_id)

                    if not (metric == 'input' and fusion == 'dynamic-mode'):
                        if res[key].mean() is np.nan:
                            # No data for the dynamic parts when doing standard fusion!
                            assert(fusion == 'NO-dynamic')
                            continue
                        elif key == best_key:
                            print(r'\textbf{{{:.4f}}}'.format(res[key].mean()), end='')
                        else:
                            print(r'        {:.4f}   '.format(res[key].mean()), end='')
                            
                        if not (metric == 'fusion' and fusion == 'dynamic-mode'):
                            print('& ', end='')
                     
            if depth == depths[0]:
                print('&', end='\n    ')
            
        print(r'\\')
        
    print("\n\n")
    for metric in metrics:
        for depth in depths:
            for fusion in fusions:
                key = "{}-{}-{}".format(metric, depth, fusion)
                acc_perc = acc_perc_agg[key]
                completeness = completeness_agg[key]
                print(key, len(acc_perc), len(completeness))
                print("Mean accuracy: {}, Mean completeness: {}".format(np.mean(acc_perc), np.mean(completeness)))

#                 box_colors.append(colors[depth][metric])
#                 columns.append(key)
#                 box_positions.append(box_offset)
    
gen_plots('../csv/tracking-res/', 'static-depth-result', eval_completeness=False)
# gen_plots('../csv/tracking-res/', 'dynamic-depth-result')


def gen_baseline_data(root, part, eval_completeness):
    print("TODO(andrei): Same as above but for the InfiniTAM baseline.")


# # Small-scale analysis of the tracklet data: does direct alignment-based pose refinement work?
# 
# Spoiler alert: No.
# 
# Takeaway: Improving the sparse methods is the way forward for the time being; track more features across more frames, and do more joint car trajectory optimization.
# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


get_ipython().magic('matplotlib inline')


import matplotlib
from matplotlib import rc
# Enable full LaTeX support in plot text. Requires a full-fledged LaTeX installation
# on your system, accessible via PATH.
# rc('text', usetex=True)

plt.rcParams["figure.figsize"] = (12, 5)
matplotlib.rcParams.update({'font.size': 16})


get_ipython().run_cell_magic('bash', '', 'ls -t ../csv/aug-21-tracklet-results/*-3d-tracking-result.csv | head -n5')


# ## What to visualize
# 
#  * Accuracy over time, aggregated over multiple vehicle tracks, say, more than 3-5 frames in length
#  * Simple bar chart showin trans/rot errors for no-da/da, possibly also labeled with the exact numbers.
# 

# ## Initial tests
# 

CSV_FILE_DN_NO_DIRECT = "../cmake-build-debug/kitti-tracking-sequence-0006-tracking-dataset-offset-0-depth-precomputed-dispnet-voxelsize-0.0500-max-depth-m-20.00-NO-direct-ref-3d-tracking-result.csv"
CSV_FILE_DN_WITH_DIRECT = "../cmake-build-debug/kitti-tracking-sequence-0006-tracking-dataset-offset-0-depth-precomputed-dispnet-voxelsize-0.0500-max-depth-m-20.00-with-direct-ref-3d-tracking-result-no-param-tweaks.csv"
CSV_FILE_DN_WITH_DIRECT_TWEAKED = "../cmake-build-debug/kitti-tracking-sequence-0006-tracking-dataset-offset-0-depth-precomputed-dispnet-voxelsize-0.0500-max-depth-m-20.00-with-direct-ref-3d-tracking-result-<todo>.csv"


dn_no_direct_frame = pd.read_csv(CSV_FILE_DN_NO_DIRECT)
dn_direct_frame = pd.read_csv(CSV_FILE_DN_WITH_DIRECT)
print(len(dn_no_direct_frame), len(dn_direct_frame))
# dn_no_direct_frame


dn_no_direct_frame.head()


dn_direct_frame.head()


# dn_no_direct_frame[dn_no_direct_frame['frame_id'] == 43]
# dn_no_direct_frame[dn_no_direct_frame['track_id'] == 2]

# dn_no_direct_frame['track_id']


def compare_tracks(no_direct_frame, direct_frame, track_id=6):
    nd_track = no_direct_frame[no_direct_frame.track_id == track_id]
    d_track = direct_frame[direct_frame.track_id == track_id]
    result = pd.DataFrame({
        'frame_id': nd_track.frame_id,
        'trans_error_no_direct': nd_track.trans_error,
        'rot_error_no_direct': nd_track.rot_error,
        'trans_error_with_direct': d_track.trans_error,
        'rot_error_with_direct': d_track.rot_error
    })
    ax = result.plot('frame_id', ['trans_error_no_direct', 'trans_error_with_direct'])
    ax.set_ylim([0, 0.5])
    ax.set_ylabel(r"$\ell_2$ norm of velocity error")
    ax.set_title("Translation error over time for track #{}".format(track_id))
    ax.legend(["No direct alignment", "With direct alignment"])
    
    ax = result.plot('frame_id', ['rot_error_no_direct', 'rot_error_with_direct'])
    ax.set_ylim([0, 0.1])
    ax.set_ylabel("Delta angle estimation error, in radians")
    ax.set_title("Rotational error over time for track #{}".format(track_id))
    ax.legend(["No direct alignment", "With direct alignment"])

compare_tracks(dn_no_direct_frame, dn_direct_frame)


# ## Aggregating multiple tracks' information, over multiple sequences.
# 

import os

def no_extreme_error(track):
    for row in track.index:
        if track['trans_error'][row] > 12.5:
            print("Too extreme!")
            return False
    
    return True


def get_tracks(df, prefix, min_track_len):
    """Returns an id->[entry] map from the given tracklet evaluation data."""
    gb = df.groupby('track_id')
    tracks = {}
    for track_id in gb.groups.keys():
        track = df[df['track_id'] == track_id]
        
        if len(track) >= min_track_len: # and no_extreme_error(track):
            track_key = '{}-{}'.format(prefix, track_id)
            track_start = track.index[0]
            new_index = track.index - track_start
            track = track.set_index(new_index)
#             print(track.index)
            tracks[track_key] = track
            
    return tracks         
    

def analyze_tracklets(root_dir, **kw):
    csv_pattern = 'kitti-tracking-sequence-{seq_id:04d}-tracking-dataset-offset-0-'         'depth-precomputed-dispnet-voxelsize-0.0500-max-depth-m-20.00-'         'dynamic-mode-{direct_ref_type}-direct-ref-NO-fusion-weights-3d-tracking-result.csv'
        
    min_track_len = kw.get('min_track_len', 10)
    seq_ids = np.arange(7)
    direct_ref_types = ['NO', 'with']
    
    data_raw = {}
    for ref_type in direct_ref_types:
        data_raw[ref_type] = {
            'track_id': []
        }
        data_raw[ref_type]['rot'] = {}
        data_raw[ref_type]['trans'] = {}
        for i in range(min_track_len):
            data_raw[ref_type]['rot']['frame-{:02d}'.format(i)] = []
            data_raw[ref_type]['trans']['frame-{:02d}'.format(i)] = []
        
    
    for direct_ref_type in direct_ref_types:
        for seq_id in seq_ids:
            csv_fname = csv_pattern.format(seq_id=seq_id, direct_ref_type=direct_ref_type)
            csv_fpath = os.path.join(root_dir, csv_fname)
            
            df = pd.read_csv(csv_fpath)
            prefix = '{}-{}'.format(seq_id, direct_ref_type)
            tracks = get_tracks(df, prefix, min_track_len)
            
            for track_key, track in tracks.items():
                data_raw[direct_ref_type]['track_id'].append(track_key)
                for i, frame_idx in enumerate(track.index[:min_track_len]):
                    data_raw[direct_ref_type]['rot']['frame-{:02d}'.format(i)].append(track['rot_error'][frame_idx])
                    data_raw[direct_ref_type]['trans']['frame-{:02d}'.format(i)].append(track['trans_error'][frame_idx])
        
#     print(np.array(data_raw).shape)
    no_da_rot = pd.DataFrame(data_raw['NO']['rot'])
    with_da_rot = pd.DataFrame(data_raw['with']['rot'])
    no_da_trans = pd.DataFrame(data_raw['NO']['trans'])
    with_da_trans = pd.DataFrame(data_raw['with']['trans'])
    
    fig, (ax_trans, ax_rot) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
    x = np.arange(2, min_track_len+2)
    MPL_BLUE = 'C0'
    MPL_ORANGE = 'C1'
    
#     ax_trans.errorbar(x, no_da_trans.mean(), yerr=no_da_trans.std(), label='No refinement (mean)')
#     ax_trans.errorbar(x, with_da_trans.mean(), yerr=with_da_trans.std(), label='With refinement (mean)')
    ax_trans.plot(x, no_da_trans.mean(), label="No refinement (mean)")
    ax_trans.plot(x, with_da_trans.mean(), label="With refinement (mean)")
    ax_trans.plot(x, no_da_trans.median(), '--', color=MPL_BLUE, label="No refinement (median)")
    ax_trans.plot(x, with_da_trans.median(), '--', color=MPL_ORANGE, label="With refinement (median)")
    
    ax_trans.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_trans.set_ylabel("Relative translation error (m)")
    ax_trans.legend()
    ax_trans.grid(color='0.7', linestyle='--')   
    ax_trans.set_title("Translation error over time")
    
    ax_rot.plot(x, no_da_rot.mean(), label="No refinement")
    ax_rot.plot(x, with_da_rot.mean(), label="With refinement")
    ax_rot.plot(x, no_da_rot.median(), '--', color=MPL_BLUE, label="No refinement (median)")
    ax_rot.plot(x, with_da_rot.median(), '--', color=MPL_ORANGE, label="With refinement (median)")
    ax_rot.legend()    
    ax_rot.grid(color='0.7', linestyle='--')   

    ax_rot.set_title("Rotation error over time")
    
    ax_rot.set_ylabel("Relative rotation error (deg)")
    ax_rot.set_xlabel("Track frame")
    ax_rot.set_xticks(np.arange(2, 2+min_track_len))
    
    assert(len(no_da_trans) == len(with_da_trans) and len(no_da_trans) == len(no_da_rot) and len(no_da_trans) == len(with_da_rot))
    print("Aggregated data from {} tracks.".format(len(no_da_trans)))
    
    fig.subplots_adjust(hspace=0.33)
    
    fig_root = os.path.join('..', 'fig')
    fig.savefig(os.path.join(fig_root, 'tracklet-results-agg.eps'))
    
    
analyze_tracklets(os.path.join('..', 'csv', 'aug-21-tracklet-results'))





