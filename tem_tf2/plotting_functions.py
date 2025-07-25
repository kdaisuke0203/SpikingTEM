#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from parameters import old2new
from model_utils import DotDict
from os import listdir, path
from scipy.signal import savgol_filter
import gzip
import copy as cp
import numpy as np
import os 
from scipy.ndimage import rotate
from scipy.signal import correlate2d
from scipy.stats import pearsonr


interpolation_method = 'None'
fontsize = 25
linewidth = 4
labelsize = 20

def compute_square_gridness(autocorr):
    center = np.array(autocorr.shape) // 2
    radius = min(center) // 2

    y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = (dist_from_center > radius) & (dist_from_center < radius + 4)

    base = autocorr.copy()

    def rotated_corr(angle):
        rotated = rotate(base, angle, reshape=False, order=1, mode='constant', cval=0.0)
        if np.std(rotated[mask]) == 0 or np.std(base[mask]) == 0:
            return np.nan
        return pearsonr(base[mask].ravel(), rotated[mask].ravel())[0]

    corr_90 = rotated_corr(90)
    corr_270 = rotated_corr(270)
    corr_45 = rotated_corr(45)

    if np.any(np.isnan([corr_90, corr_270, corr_45])):
        return np.nan

    square_gridness = (corr_90 + corr_270)/2 - corr_45
    return square_gridness

def compute_autocorr(rate_map):
    # subtract mean for zero-mean correlation
    rate_map_zero_mean = rate_map - np.nanmean(rate_map)
    return correlate2d(rate_map_zero_mean, rate_map_zero_mean, mode='full')

def compute_gridness(rate_map):
    autocorr = compute_autocorr(rate_map)
    center = np.array(autocorr.shape) // 2
    radius = 0.5  # これを調整可能
    #print("autocorr",autocorr,"center",center)

    # 中心部分のマスクを除去
    y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = (dist_from_center > radius) & (dist_from_center < radius + 5)  # 輪状マスク
    #print("mask",mask)

    angles = [30, 60, 90, 120, 150]
    scores = []

    base = autocorr * mask  # 基準（未回転）

    for angle in angles:
        rotated = rotate(autocorr, angle, reshape=False, order=1)
        rotated = rotated * mask  # 同じマスクを適用
        #print("rrrr",rotated)
        r = np.corrcoef(base[mask].ravel(), rotated[mask].ravel())[0, 1]
        #print("rrrrrrr",r)
        scores.append(r)

    # 60度/120度は六角構造に一致するので最大、それ以外は最小に
    gridness = np.min([scores[0], scores[2], scores[4]]) - np.max([scores[1], scores[3]])

    return gridness


def square_plot(cells, env, pars, plot_specs, name='sq', lims=(), mask=False, env_class=None, fig_dir=None, dates=None, runs=0):
    cell = cells[env]
    print("cells",cells.shape,"env",env,"cell",cell.shape)
    # number of cells we have
    n = np.shape(cell)[1]
    # get sub fig dimension:
    xs, ys = env_class.get_node_positions(_plot_specs=plot_specs, _mask=mask)
    print("xs, ys",xs.shape, ys.shape)
    x_dim = max(xs) - min(xs)
    y_dim = max(ys) - min(ys)
    if plot_specs.cell_num or plot_specs.max_min:
        y_dim = y_dim * 2
    # work out num cols and num rows of subplots
    if plot_specs.split_freqs:
        n = sum(plot_specs.n_cells_freq)
        # separate frequencies
        n_cols = np.argmin(
            [np.abs((np.sum([np.ceil(n_f / (i + 0.00001)) for n_f in plot_specs.n_cells_freq]) + len(
                plot_specs.n_cells_freq) - 1) * y_dim - i * x_dim) for i in range(n)])
        n_rows = np.sum([np.ceil(n_f / n_cols) for n_f in plot_specs.n_cells_freq]) + len(plot_specs.n_cells_freq) - 1
    else:
        n_cols = np.ceil(np.sqrt(n * y_dim / x_dim))
        n_rows = np.ceil(np.sqrt(n * x_dim / y_dim))

    add_on = 0
    print("number of cells",n)
    print("n_cols",n_cols,"n_rows",n_rows)
    n_cols = 18
    n_rows = 18
    path = os.path.join(os.path.dirname(os.getcwd()), "Summaries/"+dates+"/run"+runs+"/").replace("\\", "/")
    save_dirs = [path]
    """f = plt.figure(figsize=(18, 18))
    for grid in range(n):
        cell_ = cell[:, grid]

        if plot_specs.split_freqs:
            if sum(np.cumsum(plot_specs.n_cells_freq) == grid) > 0:
                add_on += n_cols if (grid + add_on) % n_cols == 0 else 2 * n_cols - ((grid + add_on) % n_cols)
            #plt.subplot(n_rows, n_cols, add_on + grid + 1)
            plt.subplot(n_rows, n_cols, grid + 1)
        else:
            plt.subplot(n_rows, n_cols, grid + 1)
        xs, ys, cell_prepared = env_class.get_node_positions(cells=cell_, _plot_specs=plot_specs, _mask=mask)
        cell_prepared = cell_prepared[:xs.shape[0]]
        #print("cell_prepared",cell_prepared.shape)
        s = plot_specs[old2new(pars.world_type)].marker_size
        marker = plot_specs[old2new(pars.world_type)].marker_shape

        s = 200 * s / (n_cols * n_rows)
        if plot_specs.cell_num:
            s = s / 1.5

        ax = plt.gca()
        ax.scatter(xs, ys, c=cell_prepared, cmap=plot_specs.cmap, s=s, marker=marker,
                   vmin=lims[0][grid] if lims else None, vmax=lims[1][grid] if lims else None)

        ax.set_xlim(np.min(xs) - 0.5, np.max(xs) + 0.5)
        ax.set_ylim(np.min(ys) - 0.5, np.max(ys) + 0.5)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xticks([])
        ax.set_yticks([])

        # make black bounding box
        ax.patch.set_edgecolor('black')
        ax.patch.set_linewidth('1')

        if plot_specs.max_min:
            maxi = max(cell_)
            mini = min(cell_)
            ax.set_title("{:.2f},{:.2f}".format(mini, maxi), {'fontsize': 20})
        if plot_specs.cell_num:
            ax.set_title(str(grid), {'fontsize': 20})
    plt.tight_layout(pad=0.95) #
    #plt.show()

    #f plot_specs.save:
    #    f.savefig((fig_dir if fig_dir else './figures/' + name) + ".png", bbox_inches='tight')
    #print("save_dirs + name)",path + name)
    f.savefig((path + name) + ".png")
    #plt.close('all')"""

    f2 = plt.figure(figsize=(18, 18))

    for grid in range(n):
        cell_ = cell[:, grid]

        if plot_specs.split_freqs:
            if sum(np.cumsum(plot_specs.n_cells_freq) == grid) > 0:
                add_on += n_cols if (grid + add_on) % n_cols == 0 else 2 * n_cols - ((grid + add_on) % n_cols)
            #plt.subplot(n_rows, n_cols, add_on + grid + 1)
            plt.subplot(n_rows, n_cols, grid + 1)
        else:
            plt.subplot(n_rows, n_cols, grid + 1)
        xs, ys, cell_prepared = env_class.get_node_positions(cells=cell_, _plot_specs=plot_specs, _mask=mask)
        cell_prepared = cell_prepared[:xs.shape[0]]
        widd = int(np.max(xs)-np.min(xs))+1
        #print("iiiii",int(np.max(xs)-np.min(xs)))
        g_map = cell_prepared.reshape(widd, widd)*10
        #print("grid",grid,"QQQQQQQQqq",g_map)
        #gridness = compute_gridness(g_map)
        #print("grid",grid,"Gridness:",gridness)
        ax = plt.gca()
        #ax2 = plt.gca()
        #print("ggggg",g_map)
        #'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           #'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           #'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
        ax.imshow(g_map, interpolation='spline16', cmap=plot_specs.cmap)
        ax.axis("off")
        ax.set_title(f"#{grid}", fontsize=12)
        #print("Gmap",g_map)
    #f plot_specs.save:
    #    f.savefig((fig_dir if fig_dir else './figures/' + name) + ".png", bbox_inches='tight')
    #print("save_dirs + name)",path + name)
    #plt.show()
    f2.savefig((path + name) + "sm.png")
    #plt.close('all')

    """f3 = plt.figure(figsize=(18, 18))

    for grid in range(n):
        cell_ = cell[:, grid]

        if plot_specs.split_freqs:
            if sum(np.cumsum(plot_specs.n_cells_freq) == grid) > 0:
                add_on += n_cols if (grid + add_on) % n_cols == 0 else 2 * n_cols - ((grid + add_on) % n_cols)
            #plt.subplot(n_rows, n_cols, add_on + grid + 1)
            plt.subplot(n_rows, n_cols, grid + 1)
        else:
            plt.subplot(n_rows, n_cols, grid + 1)
        xs, ys, cell_prepared = env_class.get_node_positions(cells=cell_, _plot_specs=plot_specs, _mask=mask)
        cell_prepared = cell_prepared[:xs.shape[0]]
        widd = int(np.max(xs)-np.min(xs))+1
        #print("iiiii",int(np.max(xs)-np.min(xs)))
        ax2 = plt.gca()
        g_map = cell_prepared.reshape(widd, widd)*10
        #print("grid",grid,"QQQQQQQQqq",g_map)
        gridness = compute_gridness(g_map)
        print("grid",grid,"Gridness:",gridness)
        #ax = plt.gca()
        #gridness_sq = compute_square_gridness(compute_autocorr(g_map))
        #print("grid",grid,"Gridness_sq:",gridness_sq)
        ax2.imshow(compute_autocorr(g_map), interpolation='spline16', cmap=plot_specs.cmap)
        ax2.axis("off")
        ax2.set_title(f"#{grid}", fontsize=12)"""

    #f plot_specs.save:
    #    f.savefig((fig_dir if fig_dir else './figures/' + name) + ".png", bbox_inches='tight')
    #print("save_dirs + name)",path + name)
    #plt.show()
    #f3.savefig((path + name) + "sm.png")
    plt.close('all')




def square_autocorr_plot(cells, env, pars, plot_specs, name='auto', env_class=None):
    width = pars.widths[env]
    cell = cells[env]

    cmap = plot_specs.cmap
    circle = plot_specs.circle
    show = plot_specs.show

    # number of cells we have
    n = np.shape(cell)[1]
    # get sub fig dimension:
    xs, ys = env_class.get_node_positions(_plot_specs=plot_specs)

    x_dim = max(xs) - min(xs)
    y_dim = max(ys) - min(ys)

    # work out num cols and num rows of subplots
    if plot_specs.split_freqs:
        # separate frequencies
        n_cols = np.argmin(
            [np.abs((np.sum([np.ceil(n_f / (i + 0.00001)) for n_f in plot_specs.n_cells_freq]) + len(
                plot_specs.n_cells_freq) - 1) * y_dim - i * x_dim) for i in range(n)])
        n_rows = np.sum([np.ceil(n_f / n_cols) for n_f in plot_specs.n_cells_freq]) + len(plot_specs.n_cells_freq) - 1
    else:
        n_cols = np.ceil(np.sqrt(n * y_dim / x_dim))
        n_rows = np.ceil(np.sqrt(n * x_dim / y_dim))

    f = plt.figure(figsize=(18, 18))
    add_on = 0
    for grid in range(n):
        # ax = plt.subplot(wid, wid, grid + 1)
        if plot_specs.split_freqs:
            if sum(np.cumsum(plot_specs.n_cells_freq) == grid) > 0:
                add_on += n_cols if (grid + add_on) % n_cols == 0 else 2 * n_cols - ((grid + add_on) % n_cols)
            plt.subplot(n_rows, n_cols, add_on + grid + 1)
        else:
            plt.subplot(n_rows, n_cols, grid + 1)

        cell_ = cell[:, grid]
        # graph-auto correlation
        auto_x, auto_y, auto_c = autocorr_with_positions(cell_, env, pars, env_class=env_class)
        auto_c_plot = cp.deepcopy(auto_c)

        if circle:
            lim = (width * 2 - 1) / 2
            radius_lim = np.minimum(np.floor(lim), np.floor(lim))
            if old2new(pars.world_type) == 'hexagonal':
                radius_lim = radius_lim * np.sqrt(3) / 2
            allowed = np.sqrt(auto_x ** 2 + auto_y ** 2) < radius_lim
            auto_c_plot[~allowed] = np.nan

        s = plot_specs[old2new(pars.world_type)].marker_size
        marker = plot_specs[old2new(pars.world_type)].marker_shape
        ax = plt.gca()
        ax.scatter(auto_x, auto_y, c=auto_c_plot, cmap=cmap, s=s, marker=marker)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=0.15)
    if show:
        plt.show()

    f.savefig('./figures/' + name + ".pdf", bbox_inches='tight')

    plt.close('all')



def get_data_path(run, date, save_dirs, recent, index=None):
    """
    Find the path where the data for the requested data and run is stored, and return the latest training iteration
    """
    for save_dir in save_dirs:
        try:
            # Build directory for this run that contains the saved data
            save_path = save_dir + date + '/run' + str(run) + '/save'
            # Find all saved data files
            list_of_files = listdir(save_path)
            print('yes ' + save_path)

            # print(list_of_files)

            # Find the most latest training iteration
            if index is None:
                index = find_most_recent(list_of_files, ['.npy', 'iter'], ['link', 'par'], recent=recent)
            else:
                index = index

            # Index is set to None if no iterations were found at all; in that case, pass
            if index is None:
                print('Run folder found, but no training iterations!')
                pass
            else:
                # If the files for the latest training iteration are within an iter folder, update path
                if any(['iter_' in file and index in file for file in list_of_files]):
                    save_path = save_path + '/iter_' + index

                # Return the save_path where the stored data is saved
                return save_path, index, list_of_files
        except FileNotFoundError:
            save_path = save_dir + date + '/run' + str(run) + '/save'
            print('not ' + save_path)
            pass

    raise ValueError('FILE NOT FOUND')


def get_model_path(run, date, save_dirs, recent=-1, index_load=None):
    """
    Find the path where the trained model weights are stored, and return the latest training iteration
    """
    for save_dir in save_dirs:
        try:
            # Build save directory for this run
            save_path = save_dir + date + '/run' + str(run)
            # Find all files in the model folder of the base directory
            list_of_files = listdir(save_path + '/model')
            print("list_of_files",list_of_files)
            if index_load is None:
                # Find the most latest training iteration
                index = find_most_recent(list_of_files, ['.index'], None, recent=recent)
            else:
                index = index_load if any([index_load in x for x in list_of_files]) else None

            # Index is set to None if no iterations were found at all; in that case, pass
            if index is None:
                print('Run folder found, but no training iterations!')
                pass
            else:
                # Return the save_dir (the base directory for storing training runs - not save_path!)
                return save_dir, index
        except FileNotFoundError:
            pass

    raise FileNotFoundError('FILE NOT FOUND')


def find_most_recent(file_list, must_contain=None, cant_contain=None, recent=-1):
    """
    Accepts a list of strings of format X_n[.Y optional], returns highest number n
    Each of the strings needs to contain one of must_contain and can't contain any of cant_contain
    """
    # Find all iteration numbers from file list where files match and sort them
    iter_numbers = [int(str(x.split('.')[0]).split('_')[-1])
                    for x in file_list
                    if (True if cant_contain is None else not any([y in x for y in cant_contain]))
                    and (True if must_contain is None else any([y in x for y in must_contain]))]
    iter_numbers.sort()

    # Index is the latest iteration, or None if no iterations were found at all
    index = None if len(iter_numbers) == 0 else str(np.unique(iter_numbers)[recent])
    return index


def load_numpy_gz(file_name):
    try:
        print("file_name",file_name)
        #file_name = 'C:/Users/kdais/prog/generalising-structural-knowledge-master/Summaries/2023-12-02/run7/save/params.npy'
        #print("file_name",file_name)
        return np.load(file_name, allow_pickle=True)
    except FileNotFoundError:
        f = gzip.GzipFile(file_name + '.gz', "r")
        return np.load(f, allow_pickle=True)


def get_data(save_dirs, run, date, recent=-1, index=None, smoothing=0, n_envs_save=None):
    """
    Load a run's timeseries, setup data, and rate maps. Lots of try-except clauses for backward compatibility:
    Previously, timeseries weren't stored, but summary statistics (like setup data and ratemaps) were.
    Alternatively, store all timeseries and the test_dict, and reconstruct all other data from those.
    """

    # Find the path where the files for this run are stored
    save_path, index, list_of_files = get_data_path(run, date, save_dirs, recent, index=index)
    print('Loading model time point ' + str(index))
    #print("save_path",save_path,os.path.dirname(save_path))
    # If files for a training iteration come in iter_[index] directories, the params file is in the parent directory
    params_path = os.path.dirname(save_path)
    #params_path = path.normpath(save_path).split(path.sep)
    #params_path_append = '/' if params_path[0] != '..' else ''
    #print("str(params_path[-1])",str(params_path[-6:-1]))
    #print("
    #params_path = params_path_append + path.join(*params_path[:-1]) if 'iter_' in str(params_path[-1]) else save_path
    #params_path = save_path#path.join(params_path[:-6]) if 'iter_' in str(params_path[-6:-1]) else save_path
    #params_path = path.join(*params_path[:-1]) 
    #print("params_path",params_path)
    # Load run parameters
    params = load_numpy_gz(params_path + '/params.npy').item()
    #print("params",params)
    params = DotDict(params)

    if n_envs_save is not None:
        print("n_envs_save",n_envs_save)
        params.n_envs_save = n_envs_save

    # Try loading run setup data
    try:
        print("save_path",save_path)
        test_dict = DotDict(load_numpy_gz(save_path + '/final_dict_' + index + '.npy').item())
        #print("test_dict",test_dict)
        #print("test_dict2",test_dict.two_hot_mat)
        adj = [x.adj for x in test_dict.curric_env.envs]
        
        # convert class params to dict
        for i, env in enumerate(test_dict.curric_env.envs):
            test_dict.curric_env.envs[i].par = DotDict(env.par)

        print('Successfully reconstructed run setup data from test_dict')
    except FileNotFoundError:
        test_dict = None
        adj = load_numpy_gz(save_path + '/adj_' + index + '.npy')
        print('Unsuccessfully reconstructed run setup data from test_dict')

    # Timeseries are numpy arrays of shape [environments (or batch size), cells, timesteps]
    g_timeseries = load_numpy_gz(save_path + '/gs_timeseries_' + index + '.npy')
    g2p_timeseries = load_numpy_gz(save_path + '/g2ps_timeseries_' + index + '.npy')
    g_gen_timeseries = load_numpy_gz(save_path + '/g_gens_timeseries_' + index + '.npy')
    g_pred2_timeseries = np.roll(np.copy(g_timeseries), -1)
    p_timeseries = load_numpy_gz(save_path + '/ps_timeseries_' + index + '.npy')
    x2p_timeseries = load_numpy_gz(save_path + '/x2ps_timeseries_' + index + '.npy')
    pos_timeseries = load_numpy_gz(save_path + '/pos_timeseries_' + index + '.npy')
    x_timeseries = load_numpy_gz(save_path + '/xs_timeseries_' + index + '.npy')
    x_gt_timeseries = load_numpy_gz(save_path + '/xs_gt_timeseries_' + index + '.npy')
    try:
        final_variables = DotDict(load_numpy_gz(save_path + '/final_variables' + index + '.npy').item())
    except FileNotFoundError:
        final_variables = None
    print('Successfully loaded timeseries')

    try:
        envs = test_dict.curric_env.envs
        #print("ENV")
    except AttributeError:
        envs = None

    print("g_timeseries.shape",g_timeseries.shape)
    np.set_printoptions(threshold=np.inf)
    #print("pos_timeseries",pos_timeseries)
    
    print("pos_timeseries",pos_timeseries.shape)
    np.set_printoptions(0)
    to_right_ind = []
    to_left_ind = []
    #for j in range(pos_timeseries.shape[1]):
    #    if j > 0:
    #        if pos_timeseries[2,j] - pos_timeseries[2,j-1] > 2:
    #            to_right_ind.append(j)
   #         else:
    #            to_left_ind.append(j)
    #g_timeseries = np.delete(g_timeseries, to_left_ind, axis=2)
    #pos_timeseries = np.delete(pos_timeseries, to_left_ind, axis=1)
    print("g_timeseries",g_timeseries.shape)
    print("pos_timeseries",pos_timeseries.shape)

    # These are 'real' ratemaps: cell activity during walk
    #print("envs",envs)
    x_all = rate_map_from_timeseries(x_timeseries, pos_timeseries, params, smoothing=smoothing, envs=envs)
    #print("x_all",len(x_all))
    g_all = rate_map_from_timeseries(g_timeseries, pos_timeseries, params, smoothing=smoothing, envs=envs)
    g2p_all = rate_map_from_timeseries(g2p_timeseries, pos_timeseries, params, smoothing=smoothing, envs=envs)
    g_gen_all = rate_map_from_timeseries(g_gen_timeseries, pos_timeseries, params, smoothing=smoothing, envs=envs)
    g_pred2_all = rate_map_from_timeseries(g_pred2_timeseries, pos_timeseries, params, smoothing=smoothing, envs=envs)
    p_all = rate_map_from_timeseries(p_timeseries, pos_timeseries, params, smoothing=smoothing, envs=envs)
    x2p_all = rate_map_from_timeseries(x2p_timeseries, pos_timeseries, params, smoothing=smoothing, envs=envs)
    # These are more like histograms, but can use the same rate-map machinery
    x_gt_timeseries = np.mean(x_gt_timeseries,axis=-1)
    correct_timeseries = np.expand_dims(np.argmax(x_gt_timeseries, axis=1) == np.argmax(x_timeseries, axis=1),
                                        1)
    acc_s_t_to = rate_map_from_timeseries(correct_timeseries, pos_timeseries, params, envs=envs)
    acc_s_t_from = rate_map_from_timeseries(correct_timeseries[:, :, 1:], pos_timeseries[:, :-1], params, envs=envs)
    positions = rate_map_from_timeseries(np.ones(correct_timeseries.shape),
                                         pos_timeseries, params, do_hist=True, envs=envs)
    print('Successfully reconstructed rate maps from timeseries')

    g_all = np.nan_to_num(g_all)
    g2p_all = np.nan_to_num(g2p_all)
    g_gen_all = np.nan_to_num(g_gen_all)
    g_pred2_all = np.nan_to_num(g_pred2_all)
    p_all = np.nan_to_num(p_all)
    x2p_all = np.nan_to_num(x2p_all)

    data = DotDict({
        'x': x_all,
        'g': g_all,
        'g2p': g2p_all,
        'g_gen': g_gen_all,
        'g_pred2': g_pred2_all,
        'p': p_all,
        'x2p': x2p_all,
        'acc_to': acc_s_t_to,
        'acc_from': acc_s_t_from,
        'positions': positions,
        'adj': adj,
        'x_timeseries': x_timeseries,
        'x_gt_timeseries': x_gt_timeseries,
        'p_timeseries': p_timeseries,
        'g_timeseries': g_timeseries,
        'pos_timeseries': pos_timeseries,
        'final_variables': final_variables
    })
    try:
        widths = params.widths
        try:
            n_states = params.n_states_world
            #print("n_sates0",n_states)
        except AttributeError:
            n_states = params.n_states
    except (AttributeError, KeyError) as e:
        widths = [x.width for x in test_dict.curric_env.envs]
        n_states = [env.n_states for env in envs]
    return data, (params, widths, n_states), list_of_files, save_path, test_dict


def sort_data(g_all, p_all, widths, plot_specs):
    # make masks
    masks = [False for _ in widths]

    if plot_specs.g_max_0:
        for i, g in enumerate(g_all):
            g_all[i] = np.maximum(g, 0)
    if plot_specs.p_max_0:
        for i, p in enumerate(p_all):
            p_all[i] = np.maximum(p, 0)
    # breakpoint()
    # max / min stuff
    g_lim = [np.nanmin(np.stack([np.nanmin(x, axis=0) for x in g_all], axis=0), axis=0),
             np.nanmax(np.stack([np.nanmax(x, axis=0) for x in g_all], axis=0), axis=0)]
    p_lim = [np.nanmin(np.stack([np.nanmin(x, axis=0) for x in p_all], axis=0), axis=0),
             np.nanmax(np.stack([np.nanmax(x, axis=0) for x in p_all], axis=0), axis=0)]

    return masks, g_lim, p_lim


def remove_zero_adj(adj_orig):
    adj = cp.deepcopy(adj_orig)
    for node in reversed(range(len(adj))):
        # if node connects to nothing and nothing connects to it, or only itself
        if (sum(adj[node]) == 0 and sum(adj[:, node]) == 0) or (sum(adj[node]) == 1 and adj[node, node] == 1):
            adj = np.delete(adj, node, 0)
            adj = np.delete(adj, node, 1)
    return adj


def tank_expand_smooth(xs, ys, cell, mult=2, smooth_val=0.1, smooth_steps=1, fill_square=False):
    xs_new = []
    ys_new = []
    cell_new = []
    adds = [[0, 0], [1, 0], [0, 1], [1, 1]]

    # expand
    for x, y, c in zip(xs, ys, cell):
        for x_add, y_add in adds:
            xs_new.append(mult * x + x_add)
            ys_new.append(mult * y + y_add)
            cell_new.append(c)

    # find new 'adjacency matrix' - connect up points with shift [0,1], [-1,1], [1,1]
    connect_points = [[0, 1], [-1, 1], [1, 1]]
    adj = np.zeros((len(xs_new), len(xs_new)))
    for i, (x, y) in enumerate(zip(xs_new, ys_new)):
        for j, (x_, y_) in enumerate(zip(xs_new, ys_new)):
            if [x_ - x, y_ - y] in connect_points:
                adj[j, i] = 1.0
    degree = np.diag(np.sum(adj, axis=1))

    # smooth
    cell_smoothed = np.asarray(cell_new)
    smooth_mat = np.matmul(degree, adj)
    for i in range(smooth_steps):
        cell_smoothed = (1.0 - smooth_val) * cell_smoothed + smooth_val * np.matmul(smooth_mat, cell_smoothed)

    cell_smoothed = list(cell_smoothed)
    xs_fin = xs_new[:]
    ys_fin = ys_new[:]
    alls = [[x, y] for x, y in zip(xs_new, ys_new)]
    if fill_square:
        # add remaining points to make square:
        for x in np.unique(xs_new):
            for y in np.unique(ys_new):
                if [x, y] not in alls:
                    xs_fin.append(x)
                    ys_fin.append(y)
                    cell_smoothed.append(0.0)

    return xs_fin, ys_fin, cell_smoothed


def autocorr_with_positions(cell, env, pars, env_class=None):
    # width = 13
    # hexy = 'hex'
    width = pars.widths[env]
    cells_ = cp.deepcopy(cell)  # g_all[env0][:, phases[0]])
    xs, ys, cells_ = env_class.get_node_positions(cells=cells_)

    radius_lim = width - 1

    xs_ys = np.concatenate([np.expand_dims(xs, 1), np.expand_dims(ys, 1)], axis=1)

    # make cross table of positions differences
    cross_table = np.expand_dims(xs_ys, 1) - np.expand_dims(xs_ys, 0)
    cross_table = np.reshape(cross_table, (-1, 2))
    _, dx_dy_indices = np.unique(np.around(cross_table, decimals=5), return_index=True, axis=0)
    dx_dy = cross_table[dx_dy_indices]
    dx_dy = dx_dy[np.sum(dx_dy ** 2, 1) <= radius_lim ** 2]

    auto_x = []
    auto_y = []
    auto_c = []
    dt = np.dtype((np.void, xs_ys.dtype.itemsize * xs_ys.shape[1]))
    for i, diff in enumerate(dx_dy):

        xs_ys_ = cp.deepcopy(xs_ys + diff)
        """
        # cross table of difference between xs_ys_ and xs_ys
        cross_table = np.expand_dims(xs_ys, 1) - np.expand_dims(xs_ys_, 0)
        ct_summed = np.sum(cross_table ** 2, 2)
        ct_summed[ct_summed < 0.00001] = 0
        orig_locs, new_locs = np.where(ct_summed == 0)
        """
        # I don't understand wy this works, but google says it does...
        orig_locs = np.nonzero(np.in1d(xs_ys.view(dt).reshape(-1), xs_ys_.view(dt).reshape(-1)))[0]
        new_locs = np.nonzero(np.in1d(xs_ys_.view(dt).reshape(-1), xs_ys.view(dt).reshape(-1)))[0]

        if len(orig_locs) < 2:
            continue

        section_1 = cells_[orig_locs]
        section_2 = cells_[new_locs]

        not_allowed = np.logical_or(np.isnan(section_1), np.isnan(section_2))

        if len(orig_locs) - np.sum(not_allowed) < 2:
            continue

        auto_x.append(diff[0])
        auto_y.append(diff[1])
        corr = np.corrcoef(section_1[~not_allowed], section_2[~not_allowed])[0][1]
        auto_c.append(corr)

        if np.isnan(np.corrcoef(section_1[~not_allowed], section_2[~not_allowed])[0][1]):
            print(i, 'isnan')

    auto_x = np.array(auto_x)
    auto_y = np.array(auto_y)
    auto_c = np.array(auto_c)

    return auto_x, auto_y, auto_c


def rate_map_from_timeseries(cell_timeseries, pos_timeseries, pars, smoothing=0, do_hist=False, envs=None):
    """
    Input cell_timeseries must be numpy matrix of shape [environments (#batches), cells, timesteps]
    If there are no cells (e.g. when calculating average occupation), expand axis 1 to have size 1
    Return ratemap: list of length #environments, containing locations by cells matrix of firing rates
    """
    #print("pars",pars.n_states)
    try:
        n_states = pars.n_states_world
    except AttributeError:
        try:
            n_states = pars.n_states
        except AttributeError:
            n_states = [env.n_states for env in envs]
            print("env", envs)

    n_cells = cell_timeseries.shape[1]
    print("")
    print("n_cells",n_cells)
    filtered = savgol_filter(cell_timeseries, smoothing + 2, 2,
                             axis=2) if smoothing else cell_timeseries
    rate_maps = []
    #print("cell_timeseries.shape",cell_timeseries.shape)
    #print("pos_timeseries.shape",pos_timeseries.size())
    #print("n_states",n_states)
    for env, (position, filt) in enumerate(zip(pos_timeseries, filtered)):
        cells_at_position = [[] for _ in range(n_states[env])]
        #print("n_states[env]",n_states[env])
        #cells_at_position = [[] for _ in range(169)]
        #print("cells_at_position",cells_at_position)
        #print("position, filt.T",position.shape, filt.T.shape)
        for pos, cells in zip(position, filt.T):
            #print("pos, cells",pos, cells)
            cells_at_position[int(pos)].append(cells)

        rate_maps.append(np.stack([(np.sum(x, axis=0) / pos_timeseries.shape[1] if do_hist else np.mean(x, axis=0))
                                   if len(x) > 0 else np.zeros(n_cells) for x in cells_at_position], axis=0))
    return rate_maps