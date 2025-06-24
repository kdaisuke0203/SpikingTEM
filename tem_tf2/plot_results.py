import numpy as np
import matplotlib.pyplot as plt
import seaborn
import datetime
import re
from os import listdir
import sys
import copy as cp
sys.path.insert(0, '../model_tf2')
import parameters
import plotting_functions as pf
import data_utils as du
import model_utils as mu
import behaviour_analyses as ba
import os
import seaborn as sns
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt

# ADD YOUR DIRECTORIES HERE
path = os.path.join(os.path.dirname(os.getcwd()), "Summaries/").replace("\\", "/")
# Choose which training run data to load
date = '2025-06-24'
run = '0'
index_load = None
save_dirs = [path]
# Try to find the most recent trained model data to run a forward pass
recent = -1
time_series_smoothing = 0
try:
    print("try")
    # Find model path and iteration index
    save_dir, index = pf.get_model_path(run, date, save_dirs, recent)
    print("index",index)
    # Run forward path for retrieved model, if folder doesn't exist yet
    model = ba.save_trained_outputs(date, run, int(index), base_path=save_dir, force_overwrite=False, n_envs_save=1)
except FileNotFoundError:
    print('No trained model weights found for ' + date + ', run ' + run + '.')
    
# Load data, generated either during training or in a forward pass through a trained model
print("Loading data")
data, para, list_of_files, save_path, env_dict = pf.get_data(save_dirs, run, date, recent, index=index, smoothing=time_series_smoothing, n_envs_save=16)

#exit()

# Unpack data
x_all = data.x
g_all = data.g
#g_pred2_all = data.g_pred2
p_all = data.p
acc_s_t_to = data.acc_to
acc_s_t_from = data.acc_from
positions = data.positions
adj = data.adj
x_timeseries = data.x_timeseries
x_gt_timeseries = data.x_gt_timeseries
p_timeseries = data.p_timeseries
print("PPPPPPPPP", p_timeseries.shape)
g_timeseries = data.g_timeseries
pos_timeseries = data.pos_timeseries
final_variables = data.final_variables
# Group timeseries together for backward compatibility
timeseries = (g_timeseries, p_timeseries, pos_timeseries)
# Assign parameters
params, widths, n_states = para

# Specify plotting parameters. Some fields will be added after loading data & parameters
plot_specs = mu.DotDict({'smoothing': 0, # spatial ratemap smoothing. Needs to be odd, or 0 for no smoothing
                      'maxmin': True,
                      'cmap': 'jet',
                      'show': True,
                      'circle': True,
                      'g_max_0': False,
                      'p_max_0': True,
                      'save': False,
                      'split_freqs': True,
                      'mult': 4,
                      'cell_num': True,
                      'rectangle': {'marker_size': 60,
                                  'marker_shape': 's'},
                     })

import seaborn
seaborn.set_style(style='white')
seaborn.set_style({'axes.spines.bottom': False,'axes.spines.left': False,'axes.spines.right': \
                   False,'axes.spines.top': False})

masks, g_lim, p_lim = pf.sort_data(g_all, p_all, widths, plot_specs)

masks = [(np.sum(g,1) + np.sum(p,1) != 0).tolist() for g,p in zip(g_all, p_all)]

#########################################################
"""
#load weights
ckpt_path = path + date + '/run' + str(run) + '/model'
print("ckpt_path",ckpt_path)

# チェックポイントを読み込む
reader = tf.train.load_checkpoint(ckpt_path)

# 変数名とその形状を表示
var_to_shape = reader.get_variable_to_shape_map()
for name in var_to_shape:
    print(f"Variable name: {name}, shape: {var_to_shape[name]}")

weights = reader.get_tensor('g2g_logsig_inf/0/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE')
print("WWW",weights)

exit()
"""
#########################
plot_specs.split_freqs = True
plot_specs.n_cells_freq = params.n_grids_all
plot_specs.cmap = 'jet'
plot_specs.node_plot = True
plot_specs.max_min = False
print("SSSS", g_all.shape, p_all.shape)
print(plot_specs.n_cells_freq)

env0 = 0 #2
env1 = 1
envs = [env0, env1]
pf.square_plot(g_all, env0, params, plot_specs, name='g0', lims=g_lim, mask=masks[env0], env_class=env_dict.curric_env.envs[env0], dates=date, runs=run)
plot_specs.n_cells_freq = params.n_place_all
plot_specs.split_freqs = False
#pf.square_plot(p_all, env0, params, plot_specs, name='p0', lims=g_lim, mask=masks[env0], env_class=env_dict.curric_env.envs[env0], dates=date, runs=run)
p_all_reshaped = p_all.reshape(1, p_all.shape[1], int(p_all.shape[2]/params.s_size_comp), params.s_size_comp)
p_all_reduced = p_all_reshaped.mean(axis=-1)
#pf.square_plot(p_all_reduced, env0, params, plot_specs, name='p0', lims=g_lim, mask=masks[env0], env_class=env_dict.curric_env.envs[env0], dates=date, runs=run)


exit()
##################################
#load weights
ckpt_path = path + date + '/run' + str(run) + '/model'
print("ckpt_path",ckpt_path)

# チェックポイントを読み込む
reader = tf.train.load_checkpoint(ckpt_path)

# 変数名とその形状を表示
var_to_shape = reader.get_variable_to_shape_map()
for name in var_to_shape:
    print(f"Variable name: {name}, shape: {var_to_shape[name]}")

# inputs to EC
weights_p2g = reader.get_tensor('p2g_mu/0/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE')
#weights_g2g = reader.get_tensor('g2g_logsig_inf/0/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE')
#weights_t_vec = reader.get_tensor('t_vec/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE')
# g2g transition: t_mat

print("WWW",weights_p2g.shape)

a = weights_p2g[:,0:1]

# グラフの作成
G = nx.Graph()

# ノードの追加（層ごとにまとめる）
input_nodes = list(range(1, a.shape[0] + 1))
hidden_nodes = list(range(a.shape[0] + 1, a.shape[1] + a.shape[0] + 1))
#print("input_nodes",input_nodes)
G.add_nodes_from(input_nodes)
G.add_nodes_from(hidden_nodes)

# エッジの追加（仮に全結合とする）
for i in  range(a.shape[0]):
    for j in range(a.shape[1]):
        G.add_edge(i+1, j+a.shape[0]+1, weight=a[i][j])

# ニューラルネット風の位置設定（手動で x, y を指定）
pos ={}
for i in range(1, a.shape[0] + 1):
    pos[i] = (0, -i)
for j in range(a.shape[0] + 1, a.shape[1] + a.shape[0] + 1):
    pos[j] = (2, -j+a.shape[0]-5)

# 描画
plt.figure(figsize=(6, 4))
nx.draw(G, pos, with_labels=True, node_size=100, node_color='skyblue', font_size=12)
# エッジの重みを取得し、太さに変換（適当に倍率をかける）
weights = nx.get_edge_attributes(G, 'weight')
edge_widths = [weights[edge]*5 for edge in G.edges()]
nx.draw_networkx_edges(G, pos, width=edge_widths)
#nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True)})
plt.title("Neural Network Style Graph")
plt.axis('off')
plt.show()