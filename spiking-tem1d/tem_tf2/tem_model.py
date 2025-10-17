import model_utils
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from snn_layers import *
import poisson_spike

eps = model_utils.eps



class SimpleSNN(tf.keras.Model):
    def __init__(self, par, output_size, nn_type, time_steps=1):
        super().__init__()
        self.time_steps = time_steps
        if nn_type == 'sensory':
            self.fc1 = SpikingDense0(par, output_size=output_size,name='MLP_c_spike_1')
            #self.fc2 = SpikingDense(input_size=hidden_size, output_size=output_size,name='MLP_c_spike_2')
        if nn_type == 'p2g':
            self.fc1 = SpikingDense0(par, output_size=output_size,name='p2g_spike_1')
        if nn_type == 'x2p':
            #self.fc1 = SpikingDense(par, output_size=output_size,name='x2p_spike_1')
            self.fc1 = SpikingDense0(par, output_size=output_size,name='x2p_spike_1')
        if nn_type == 'x2dg':
            #self.fc1 = SpikingDense(par, output_size=output_size,name='x2p_spike_1')
            self.fc1 = SpikingDense0(par, output_size=output_size,name='x2dg_spike_1')
        if nn_type == 'g2g':
            self.fc1 = SpikingDense(par, output_size=output_size,name='g2g_spike_1')
            #self.fc2 = SpikingDense(input_size=hidden_size, output_size=output_size,name='g2g_spike_2')
        if nn_type == 'infer_p':
            self.fc1 = SpikingDense(par, output_size=output_size,name='infer_p_spike_1')
        if nn_type == 'dg':
            self.fc1 = SpikingDense0(par, output_size=output_size,name='dg_spike_1')
        if nn_type == 'dg2':
            self.fc1 = SpikingDense0(par, output_size=output_size,name='dg_spike_2')
        if nn_type == 'ca3':
            self.fc1 = SpikingDense(par, output_size=output_size,name='ca3_spike_1')
        if nn_type == 'infer_g':
            self.fc1 = SpikingDense(par, output_size=output_size,name='infer_g_spike_1')
        if nn_type == 'x2lec':
            #self.fc1 = SpikingDense(par, output_size=output_size,name='x2p_spike_1')
            self.fc1 = SpikingDense0(par, output_size=output_size,name='x2lec_spike_1')
        if nn_type == 'p2g_gen':
            self.fc1 = SpikingDense0(par, output_size=output_size,name='p2g_gen_spike_1')
        if nn_type == 'g2x_gen':
            self.fc1 = SpikingDense0(par, output_size=output_size,name='g2x_gen_spike_1')
        if nn_type == 'gen_p':
            self.fc1 = SpikingDense0(par,  output_size=output_size,name='gen_p_spike_1')

    def call(self, inputs):
        x, v = inputs  # 入力を unpack
        #print("XXXXXXXXXXXVVVVVVVVVVVV",x,v)
        h, v1 = self.fc1((x, v))    # v: 初期電位
        #print("HHHHHHHHHHHHH",h)
        #o, v2 = self.fc2((h, v1))   # v1: 中間層の膜電位
        return h, v1 

class TEM(tf.keras.Model):
    def __init__(self, par):
        super(TEM, self).__init__()

        self.par = par
        self.spike_step = par.spike_windows
        self.precision = tf.float32 if 'precision' not in self.par else self.par.precision
        self.mask = tf.constant(par.mask_p, dtype=self.precision, name='mask_p')
        self.mask_g = tf.constant(par.mask_g, dtype=self.precision, name='mask_g')
        self.batch_size = self.par.batch_size
        self.scalings = None  # JW: probs need to change this
        self.seq_pos = tf.zeros(self.batch_size, dtype=self.precision, name='seq_pos')
        if 'two_hot_mat' in par:
            self.two_hot_mat = tf.constant(par.two_hot_mat, dtype=self.precision, name='two_hot_mat')

        # Create trainable parameters
        glorot_uniform = tf.keras.initializers.GlorotUniform()
        trunc_norm_p2g = tf.initializers.TruncatedNormal(stddev=self.par.p2g_init)
        trunc_norm_g = tf.initializers.TruncatedNormal(stddev=self.par.g_init)

        # filtering constants
        self.gamma = [
            tf.Variable(np.log(self.par.freqs[f] / (1 - self.par.freqs[f])), dtype=self.precision, trainable=True,
                        name='gamma_' + str(f)) for f in range(self.par.n_freq)]
        # Entorhinal preference weights
        #self.w_x = tf.Variable(1.0, dtype=self.precision, trainable=True, name='w_x')
        #self.p_p = tf.Variable(0.5, dtype=self.precision, trainable=True, name='p_p')
        #self.p2g = tf.Variable(0.5, dtype=self.precision, trainable=True, name='p2g')
        #self.g_g = tf.Variable(0.1, dtype=self.precision, trainable=True, name='g_g')
        self.g_spike = tf.Variable(0.1, dtype=self.precision, trainable=True, name='g_spike')
        #self.ca3_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='ca3_spike')
        #self.ca1_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='ca1_spike')
        #self.stdp_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='stdp_spike')
        #self.x2g_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='x2g_spike')
        #self.x2dg_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='x2dg_spike')
        #self.g2_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='g2_spike')
        #self.g3_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='g2_spike')
        #self.g2ca3_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='g2ca3_spike')
        self.g2g_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='g2g_spike')
        self.gen_p_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='gen_p_spike')
        #self.dg_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='dg_spike')
        #self.dg2_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='dg2_spike')
        self.p_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='p_spike')
        #self.p_gt_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='p_gt_spike')
        #self.g2p_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='g2p_spike')
        #self.x_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='x_spike')
        #self.fx_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='fx_spike')
        #self.fx2_spike = tf.Variable(1.0, dtype=self.precision, trainable=True, name='fx2_spike')        
        # Entorhinal preference bias
        #self.b_x = tf.Variable(tf.zeros_initializer()(shape=self.par.s_size_comp, dtype=self.precision), trainable=True, name='bias_x')
        # Frequency module specific scaling of sensory experience before input to hippocampus
        #self.w_p = [tf.Variable(1.0, dtype=self.precision, trainable=True, name='w_p_' + str(f)) for f in range(self.par.n_freq)]

        """self.m_g2g = self.add_weight(
            #shape=(1, self.par.g_size * self.par.k),
            shape=(1, self.par.g_size),
            initializer='random_uniform',
            trainable=True,
            name='m_g2g_state'
        )
        
        self.m_x2g = self.add_weight(
            #shape=(1, self.par.g_size * self.par.k),
            shape=(1, self.par.g_size),
            initializer='random_uniform',
            trainable=True,
            name='m_x2g_state'
        )
        self.m_inf_p = self.add_weight(
            #shape=(1, self.par.g_size * self.par.k),
            shape=(1, self.par.p_size),
            initializer='random_uniform',
            trainable=True,
            name='m_inf_p_state'
        )
        self.m_gen_p = self.add_weight(
            #shape=(1, self.par.g_size * self.par.k),
            shape=(1, self.par.p_size),
            initializer='random_uniform',
            trainable=True,
            name='m_gen_p_state'
        )"""
        self.v_p2g = self.add_weight(
            #shape=(1, self.par.g_size * self.par.k),
            shape=(1, self.par.g_size),
            initializer='zeros',
            trainable=False,
            name='v_p2g_state'
        )
        self.v_p2g_gen = self.add_weight(
            #shape=(1, self.par['g_size'] * self.par['k']),
            shape=(1, self.par['g_size']),
            initializer='zeros',
            trainable=False,
            name='v_p2g_gen_state'
        )
        self.v_p_g_gen = self.add_weight(
            #shape=(1, self.par['g_size'] * self.par['k']),
            shape=(1, self.par['g_size']),
            initializer='zeros',
            trainable=False,
            name='v_p_g_gen_state'
        )
        self.v_g2x_gen = self.add_weight(
            #shape=(1, self.par['g_size'] * self.par['k']),
            shape=(1, self.par['g_size']),
            initializer='zeros',
            trainable=False,
            name='v_g2x_gen_state'
        )
        self.v_gt2x_gen = self.add_weight(
            #shape=(1, self.par['g_size'] * self.par['k']),
            shape=(1, self.par['g_size']),
            initializer='zeros',
            trainable=False,
            name='v_gt2x_gen_state'
        )
        self.v_p2gt_gen = self.add_weight(
            #shape=(1, self.par['g_size'] * self.par['k']),
            shape=(1, self.par['g_size']),
            initializer='zeros',
            trainable=False,
            name='v_p2gt_gen_state'
        )
        self.v_x2lec = self.add_weight(
            shape=(1, self.par['g_size']),
            #shape=(1, self.par['p_size'] * self.par['k']),
            initializer='zeros',
            trainable=False,
            name='v_x2lec_state'
        )
        self.v_x2p = self.add_weight(
            shape=(1, self.par['p_size']),
            #shape=(1, self.par['p_size'] * self.par['k']),
            initializer='zeros',
            trainable=False,
            name='v_x2p_state'
        )
        self.v_x2dg = self.add_weight(
            shape=(1, self.par['g_size']),
            #shape=(1, self.par['p_size'] * self.par['k']),
            initializer='zeros',
            trainable=False,
            name='v_x2dg_state'
        )
        self.v_infer_p = self.add_weight(
            #shape=(1, self.par['p_size'] * self.par['k']),
            shape=(1, self.par['p_size']),
            initializer='zeros',
            trainable=False,
            name='v_infer_p_state'
        )
        self.v_infer_g2 = self.add_weight(
            #shape=(1, self.par['p_size'] * self.par['k']),
            shape=(1, self.par['g_size']),
            initializer='zeros',
            trainable=False,
            name='v_infer_g_state'
        )
        self.v_gen_p2 = self.add_weight(
            shape=(1, self.par['p_size']),
            initializer='zeros',
            trainable=False,
            name='v_gen_p2_state'
        )
        self.v_gen_p = self.add_weight(
            shape=(1, self.par['p_size']),
            initializer='zeros',
            trainable=False,
            name='v_gen_p_state'
        )
        self.v_g2g = self.add_weight(
            #shape=(1, self.par['g_size'] * self.par['k']),
            shape=(1, self.par['g_size']),
            initializer='zeros',
            trainable=False,
            name='v_g2g_state'
        )
        self.v_dg = self.add_weight(
            #shape=(1, self.par['p_size'] * self.par['k']),
            shape=(1, self.par['dg_size']),
            initializer='zeros',
            trainable=False,
            name='v_dg_state'
        )
        self.v_dg2 = self.add_weight(
            #shape=(1, self.par['p_size'] * self.par['k']),
            shape=(1, self.par['dg_size']),
            initializer='zeros',
            trainable=False,
            name='v_dg3_state'
        )
        self.v_ca3 = self.add_weight(
            #shape=(1, self.par['p_size'] * self.par['k']),
            shape=(1, self.par['p_size']),
            initializer='zeros',
            trainable=False,
            name='v_ca3_state'
        )
        self.v_fx = self.add_weight(
            shape=(1, self.par.s_size),
            initializer='zeros',
            trainable=False,
            name='v_fx_state'
        )
        self.v_fx_pg = self.add_weight(
            shape=(1, self.par.s_size),
            initializer='zeros',
            trainable=False,
            name='v_fxpg_state'
        )
        self.v_fx_pgt = self.add_weight(
            shape=(1, self.par.s_size),
            initializer='zeros',
            trainable=False,
            name='v_fxpgt_state'
        )
        self.stdp_W = tf.Variable(tf.random.uniform((self.par['p_size'], self.par['p_size']), -0.05, 0.05), trainable=False, dtype=tf.float32)
        
        # g_prior mu
        self.g_prior_mu = tf.Variable(trunc_norm_g(shape=(1, self.par.g_size), dtype=self.precision), trainable=True,
                                      name='g_prior_mu')
        # g_prior logsig
        #self.g_prior_logsig = tf.Variable(trunc_norm_g(shape=(1, self.par.g_size), dtype=self.precision), trainable=True, name='g_prior_logsig')

        self.g_init = None

        # MLP for transition weights
        self.t_vec = tf.keras.Sequential([Dense(self.par.d_mixed_size, input_shape=(self.par.n_actions,),
                                                activation=tf.tanh, kernel_initializer=glorot_uniform, name='t_vec_1',
                                                use_bias=True), Dense(self.par.g_size ** 2, use_bias=True,
                                                                       kernel_initializer=tf.zeros_initializer,
                                                                       name='t_vec_2')])

        # p2g
        if 'p' in self.par.infer_g_type:
            """self.p2g_mu = [tf.keras.Sequential([Dense(2 * g_size, input_shape=(phase_size,), activation=tf.nn.elu,
                                                      name='p2g_mu_1_' + str(f), kernel_initializer=glorot_uniform),
                                                Dense(g_size, name='p2g_mu_2_' + str(f),
                                                      kernel_initializer=trunc_norm_p2g)]) for f, (g_size, phase_size)
                           in enumerate(zip(self.par.n_grids_all, self.par.n_phases_all))]"""
            """self.p2g_mu = [tf.keras.Sequential([Dense(g_size, input_shape=(phase_size,), activation=tf.nn.elu,
                                                      name='p2g_mu_1_' + str(f), kernel_initializer=glorot_uniform),
                                                ]) for f, (g_size, phase_size)
                           in enumerate(zip(self.par.n_grids_all, self.par.n_phases_all))]
            self.p2g_mu0 = tf.keras.Sequential([Dense(int(self.par.g_size/5), input_shape=(self.par.p_size,), activation=tf.nn.elu,
                                                      name='p2g_mu_1_', kernel_initializer=glorot_uniform),
                                                Dense(self.par.g_size, name='p2g_mu_2_',
                                                      kernel_initializer=trunc_norm_p2g)
                                                ])"""
            """self.p2g_mu_spiking = [Transition_Model(self.par, 'p2g_mu', fre=f, g_size=g_size) for f, (g_size, phase_size)
                           in enumerate(zip(self.par.n_grids_all, self.par.n_phases_all))]"""
            self.p2g_mu_spiking2 = [SimpleSNN(self.par, g_size, nn_type='p2g') for f, (g_size, phase_size)
                           in enumerate(zip(self.par.n_grids_all, self.par.n_phases_all))]

            #self.p2g_mu_spiking3 = SimpleSNN(self.par, self.par.p_size, 2*self.par.g_size, self.par.g_size, nn_type='p2g')

            """self.p2g_logsig = [tf.keras.Sequential([Dense(1 * g_size, input_shape=(2,), activation=tf.nn.elu,
                                                          kernel_initializer=glorot_uniform,
                                                          name='p2g_logsig_1_' + str(f)),
                                                    Dense(g_size, kernel_initializer=glorot_uniform, activation=tf.tanh,
                                                          name='p2g_logsig_2_' + str(f))]) for f, g_size in
                               enumerate(self.par.n_grids_all)]"""

        #g2p
        """self.g2p_mu = tf.keras.Sequential([Dense(self.par.g_size, input_shape=(self.par.g_size,),
                                                     activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                     name='g2p_1'),
                                            Dense(self.par.p_size, 
                                                      name='g2p_2', kernel_initializer=glorot_uniform),
                                                ])"""

        #g2g
        self.g2g_mu_spike = SimpleSNN(self.par, self.par.g_size,time_steps=1, nn_type='g2g')
        # g2g logsigs
        self.x2p_spiking = SimpleSNN(self.par, self.par['p_size'], nn_type='x2p')
        self.x2dg_spiking = SimpleSNN(self.par, self.par['g_size'], nn_type='x2dg')
        self.infer_p_spiking = SimpleSNN(self.par, self.par['p_size'], nn_type='infer_p')
        self.infer_g_spiking2 = SimpleSNN(self.par, self.par['g_size'], nn_type='infer_g')
        self.gen_p_spiking = SimpleSNN(self.par, self.par['p_size'], nn_type='gen_p')
        self.dg_spiking = SimpleSNN(self.par, self.par['dg_size'], nn_type='dg')
        self.dg_spiking2 = SimpleSNN(self.par, self.par['dg_size'], nn_type='dg2')
        self.ca3_spiking = SimpleSNN(self.par, self.par['p_size'], nn_type='ca3')
        self.x2lec_spiking = SimpleSNN(self.par, self.par['g_size'], nn_type='x2lec')
        self.p2g_gen_spiking = SimpleSNN(self.par, self.par['g_size'], nn_type='p2g_gen')
        #self.g2x_gen_spiking = SimpleSNN(self.par, self.par['g_size'], nn_type='g2x_gen')
        #self.torus_2d = tf.keras.layers.Dense(2, name='torus')
        
        # MLP for compressing sensory observation
        """if not self.par.two_hot:
            self.MLP_c = tf.keras.Sequential([Dense(self.par.s_size_comp_hidden, input_shape=(self.par.s_size,),
                                                    activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                    name='MLP_c_1'),
                                              Dense(self.par.s_size_comp, kernel_initializer=glorot_uniform,
                                                    name='MLP_c_2')])
            self.MLP_c = tf.keras.Sequential([Dense(self.par.s_size_comp, input_shape=(self.par.s_size,),
                                                    activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                    name='MLP_c_1')])

            self.MLP_c_star = tf.keras.Sequential([Dense(self.par.s_size_comp_hidden, input_shape=(self.par.s_size_comp,),
                                                     activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                     name='MLP_c_star_1'),
                                               Dense(self.par.s_size, kernel_initializer=glorot_uniform,
                                                     name='MLP_c_star_2')])
            self.MLP_c_star = tf.keras.Sequential([Dense(self.par.s_size, input_shape=(self.par.s_size_comp,),
                                                     activation=tf.nn.elu, kernel_initializer=glorot_uniform,
                                                     name='MLP_c_star_1')])"""
        #self.MLP_c_star_spiking = Transition_Model(self.par, 'sensory_star')
        self.MLP_c_star_spiking2 = SimpleSNN(self.par,self.par.s_size,time_steps=1, nn_type='sensory')

        self.ca1_phase_his = np.zeros((self.par['p_size'], self.par['num_states'], self.par['spike_windows']))

    @model_utils.define_scope
    def call(self, inputs, training=None, mask=None):

        # inputs = model_utils.copy_tensor(inputs_)
        # Setup member variables and get dictionaries from input
        variable_dict = self.init_input(inputs)

        # Precompute transitions
        ta_mat = self.precomp_trans(inputs.d)

        # book-keeping
        g_t, x_t, ca3_prev, p2g_prev , g_spike, ca1_prev = inputs.g, inputs.x_, inputs.ca3_prev, inputs.p2g_prev, inputs.g_prev, inputs.ca1_prev
        #g_spike = g_t[..., tf.newaxis]
        #g_spike = tf.tile(g_spike, [1, 1, self.spike_step])  
        #g_spike = self.poisson_spike(g_spike*self.g_spike)

        #ca3_prev = tf.zeros([1, self.par['p_size'], self.spike_step])  
        for i in tf.range(self.par.seq_len, name='iteration') if self.par.tf_range else range(self.par.seq_len):

            # single step
            g_t, x_t, variable_dict, g_spike, ca3_prev, p2g_prev, ca1_prev = self.step(inputs, g_t, x_t, variable_dict, i, ta_mat.read(i), g_spike, ca3_prev, p2g_prev, ca1_prev)
            #g_t, x_t, variable_dict, memories_dict = self.step(inputs, g_t, x_t, variable_dict, memories_dict, i, t_mat=0)

        # Now do full hebbian matrices update after BPTT truncation
        hebb_mat, hebb_mat_inv = inputs.hebb_mat, inputs.hebb_mat_inv #self.final_hebbian(inputs.hebb_mat, inputs.hebb_mat_inv, memories_dict)
        #print("################")

        # convert tensorarray to list
        variable_dict = self.tensorarray_2_list(variable_dict)
        variable_dict['weights']['gamma'] = self.gamma
        #print("variable_dict",variable_dict)
        #print("hebb_mat",hebb_mat)

        # Collate x_s, g for re-input to model
        re_input_dict = model_utils.DotDict({'a_rnn': hebb_mat,
                                             'a_rnn_inv': hebb_mat_inv,
                                             'x_s': tf.concat(x_t, axis=1, name='x_s_concat'),
                                             'g': g_t,
                                             'ca3_prev': ca3_prev,
                                             'ca1_prev': ca1_prev,
                                             'p2g_prev': p2g_prev,
                                             'g_prev': g_spike
                                             })

        return variable_dict, re_input_dict

    # WRAPPER FUNCTIONS

    @model_utils.define_scope
    def step(self, inputs, g_t, x_t, variable_dict, i, t_mat, g_t_spike, ca3_prev, p2g_prev, ca1_prev, mem_offset=0):
        # with tf.range and in graph mode, can't make the 'i' variable a global. So pass seq_pos, and i
        seq_pos = inputs.seq_i * self.par.seq_len + tf.cast(i, self.precision)


        # generative transition
        g_gen, g2g_spike = self.gen_g(g_t, t_mat, seq_pos, inputs.d[i], g_t_spike)

        # infer hippocampus (p) and entorhinal (g)
        #mem_inf = self.mem_step(memories_dict, 'inf', i + mem_offset)
        g, p, x_s, p_x, p_spike, dg_all, ca3_all, g_spike, x2p_spike, dg_all2= self.inference(g_gen, inputs.x[i], inputs.x_two_hot[i], x_t, inputs.d[i], g2g_spike, ca3_prev, p2g_prev, g_t_spike, inputs.positions[i])
        #tf.print("i",i,"g11111", g[0], summarize=-1)
        # generate sensory
        #mem_gen = self.mem_step(memories_dict, 'gen', i + mem_offset)
        x_all, x_logits_all, p_g, p_gt_spike, p2g_all = self.generation(p, g_spike, g_gen, p_spike, g2g_spike, x2p_spike)

        if self.par['stdp']:
          #self.stdp_update(p_gt_spike, p_spike, self.stdp_W)
          self.stdp_update2(p_gt_spike, p_spike, self.stdp_W)

        # Hebbian update - equivalent to the matrix updates, but implemented differently for computational ease
        #memories_dict = self.hebbian(p, p_g, p_x, memories_dict, i + mem_offset)

        # Collate all variables for losses and saving representations
        var_updates = [[['p', 'p'], p],
                       [['p', 'p_g'], p_g],
                       [['p', 'p_x'], p_x],
                       [['p', 'dg'], tf.reduce_mean(dg_all, axis=2)+tf.reduce_mean(dg_all2, axis=2)],
                       [['p', 'ca3'], tf.reduce_mean(ca3_all, axis=2)],
                       [['g', 'g'], g],
                       [['g', 'g_2d'], g_t],
                       [['g', 'g_gen'], g_gen],
                       [['g_prev'], g_spike],
                       [['g2g_spike'], g2g_spike],
                       [['x_s'], x_s],
                       [['ca3_prev'], ca3_prev],
                       [['ca1_prev'], ca1_prev],
                       [['p2g_prev'], p2g_all],
                       [['pred', 'x_p'], x_all['x_p']],
                       [['pred', 'x_g'], x_all['x_g']],
                       [['pred', 'x_gt'], x_all['x_gt']],
                       [['logits', 'x_p'], x_logits_all['x_p']],
                       [['logits', 'x_g'], x_logits_all['x_g']],
                       [['logits', 'x_gt'], x_logits_all['x_gt']],
                       ]

        # And write all variables to tensorarrays
        variable_dict = self.update_vars(variable_dict, var_updates, i)

        return g, x_s, variable_dict, g_spike, ca3_all, p2g_all, p_spike

    @model_utils.define_scope
    def inference(self, g_gen, x, x_two_hot, x_, d, g2g_spike, ca3_prev, p2g_prev, g_t_spike, pos):
        """
        Infer all variables
        """
        # get sensory input to hippocampus
        x2p, x_s, x2p_spike, x_spike, x2dg_all = self.x2p(x, x_, x_two_hot, d)
        x2g_spike = self.x2g(x, x_, x_two_hot, d)

        # infer entorhinal
        g, p_x, g_spike = self.infer_g(g_gen, x2p, x, g2g_spike, x2g_spike, p2g_prev, g_t_spike)

        # infer hippocampus
        p, p_spike, dg_all, ca3_all, dg_all2, ca3_prev = self.infer_p(x2p, g, x2p_spike, g_spike, x2dg_all, ca3_prev, pos)

        return g, p, x_s, p_x, p_spike, dg_all, ca3_all, g_spike, x2p_spike, dg_all2

    @model_utils.define_scope
    def generation(self, p, g_spike, g_gen, p_spike, g2g_spike, x2p_spike):
        
        x_p, x_p_logits, p2g_all = self.f_x(p_spike, "p")

        #p_gt = self.gen_p(g_gen, memories)
        v = self.v_gen_p
        mu_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        #v2 = self.v_gen_p2
        #mu_all2 = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        for i in range(self.spike_step):
            #o, v = self.gen_p_spiking((tf.concat([g2g_spike, x2p_spike], axis=1)[:,:,i], v))
            #o, v = self.infer_g_spiking((tf.concat([tf.stop_gradient(g2g_spike), p2g_all], axis=1)[:,:,i], v))
            #o, v = self.infer_g_spiking((tf.concat([g2g_spike, tf.stop_gradient(p2g_all)], axis=1)[:,:,i], v))
            #o, v = self.gen_p_spiking((g2g_spike[:,:,i]*self.gen_p_spike, v))
            o, v = self.gen_p_spiking((g2g_spike[:,:,i], v))
            #mu_all = mu_all.write(i, o*self.m_gen_p)
            mu_all = mu_all.write(i, o)
            #o2, v2 = self.gen_p_spiking((g_spike[:,:,i], v2))
            #mu_all2 = mu_all2.write(i, o2)
        self.v_gen_p.assign(v)
        mu_all = mu_all.stack() 
        p_gt_spike = tf.transpose(mu_all, perm=[1, 2, 0]) 
        p_gt = tf.reduce_mean(p_gt_spike, axis=2)

        #self.v_gen_p2.assign(v2)
        #mu_all2 = mu_all2.stack() 
        #p_g_spike = tf.transpose(mu_all2, perm=[1, 2, 0]) 
        #p_g = tf.reduce_mean(p_g_spike, axis=2)

        if self.par['stdp']:
            p_gt_spike = self.lif_stdp_output(p_gt_spike, self.stdp_W)
            p_gt_spike = tf.expand_dims(p_gt_spike, axis=0)
            p_gt = tf.reduce_mean(p_gt_spike, axis=2)
        x_gt, x_gt_logits, _ = self.f_x(p_gt_spike, "p_gt")

        #p_g = p_gt #self.gen_p(g, memories)
        x_g, x_g_logits = x_gt, x_gt_logits #self.f_x(p_g_spike, "p_g")

        x = model_utils.DotDict({'x_p': x_p,
                                 'x_g': x_g,
                                 'x_gt': x_gt})
        x_logits = model_utils.DotDict({'x_p': x_p_logits,
                                        'x_g': x_g_logits,
                                        'x_gt': x_gt_logits})

        return x, x_logits, p_gt, p_gt_spike, p2g_all

    @model_utils.define_scope
    def infer_g(self, g_gen, mu_x2p, x, g2g_spike, x2g_spike, p2g_prev, g_t_spike):

        #p_x = None
        p_x = mu_x2p #self.attractor(mu_x2p, memories)
        #mu = g_gen

        #mu_p2g, sigma_p2g, p_x = self.p2g(mu_x2p, x, memories)
        #mu = g_gen + tf.reduce_mean(x2g_spike, axis=2) *self.p_p + tf.reduce_mean(p2g_prev, axis=2) *self.p2g#+ tf.reduce_mean(g_t_spike, axis=2) *self.g_g
        #g_spike = tf.tile(tf.expand_dims(mu, axis=2), multiples=[1,1,self.spike_step]) 
        #g_spike = self.poisson_spike(g_spike*self.g_spike)
        #g_spike = g2g_spike + x2g_spike*self.p_p #+ g_t_spike*self.g_g#+ p2g_prev*self.p2g
        #mu = tf.reduce_mean(g_spike, axis=2)#*self.g2_spike
        
        #mu = g_gen + x2g *self.p_p
        #_, mu, _, sigma = model_utils.combine2(mu, mu_p2g, sigma, sigma_p2g, self.batch_size)

        ###################################
        #g2g_spike = tf.tile(tf.expand_dims(mu, axis=2), multiples=[1,1,self.spike_step]) #*self.g2g_spike
        #g2g_spike = self.poisson_spike(g2g_spike*self.g2g_spike)#*self.g_spike
        #p2g_all = tf.tile(tf.expand_dims(mu_p2g, axis=2), multiples=[1,1,self.spike_step*5]) 
        #p2g_all = self.poisson_spike(p2g_all)
        #x_spike = tf.tile(tf.expand_dims(x, axis=2), multiples=[1,1,self.spike_step]) #*self.x_spike
        #x_spike = self.poisson_spike(x_spike)

        ##########################################
        v = self.v_infer_g2
        mu_all2 = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        #v_all2 = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        for i in range(self.spike_step):
            #o, v = self.infer_g_spiking2((g_spike_[:,:,i], v))
            #o, v = self.infer_g_spiking2((tf.concat([g2g_spike*self.g_spike, x2g_spike*self.g2_spike, g_t_spike*self.g3_spike], axis=1)[:,:,i], v))
            o, v = self.infer_g_spiking2((tf.concat([g2g_spike, x2g_spike, p2g_prev], axis=1)[:,:,i], v))
            #o, v = self.infer_g_spiking2((tf.concat([g2g_spike, x2g_spike*self.x2g_spike, g_t_spike], axis=1)[:,:,i], v))
            #o, v = self.infer_g_spiking2((tf.concat([g2g_spike, p2g_all], axis=1)[:,:,i], v))
            #o, v = self.infer_g_spiking2((tf.concat([g2g_spike, x_spike], axis=1)[:,:,i], v))
            #o, v = self.infer_g_spiking2((tf.concat([g2g_spike, g_spike_], axis=1)[:,:,i], v))
            #o = g2g_spike[:,:,i] + self.p_p*p2g_all[:,:,i]
            #v_sum.append(tf.where(o == 1, tf.ones_like(v)*self.par['thr'], v))
            #v_sum.append(tf.where(o == 1, tf.ones_like(v), tf.zeros_like(v)))
            mu_all2 = mu_all2.write(i, o*self.g_spike)
            #v_all2 = v_all2.write(i, tf.where(o == 1, tf.ones_like(v)*1.0, v))
        #tf.print("GGG",self.g2_spike,self.g3_spike)
        self.v_infer_g2.assign(v)
        mu_all2 = mu_all2.stack()  # shape: (spike_step, batch_size, p_size)
        #v_all2 = v_all2.stack()
        g_spike = tf.transpose(mu_all2, perm=[1, 2, 0])
        #v_sum= tf.transpose(v_all2, perm=[1, 2, 0]) 
        #g_spike = tf.reshape(g_spike, (self.par['g_size'],self.par['k'], self.spike_step))
        #v_sum = tf.reshape(v_sum, (self.par['g_size'],self.par['k'], self.spike_step))
        #random_indices= tf.random.uniform(shape=(self.par['g_size'],), minval=0, maxval=self.par['k'], dtype=tf.int32)
        #batch_indices = tf.range(self.par['g_size'], dtype=tf.int32)
        #gather_indices = tf.stack([batch_indices, random_indices], axis=1)  
        #g_spike = tf.gather_nd(g_spike, gather_indices)
        #g_spike = tf.reshape(g_spike, (1, self.par['g_size'], self.spike_step,))
        #v_sum = tf.gather_nd(v_sum, gather_indices)
        #v_sum = tf.reshape(v_sum, (1, self.par['g_size'], self.spike_step,))
        mu = tf.reduce_mean(g_spike, axis=2)
        #g_2d = mu #self.torus_2d(mu)

        return mu, p_x, g_spike

    @model_utils.define_scope
    def infer_p(self, x2p, g, x2p_all, g_spike, x2dg_all, ca3_prev, pos):

        # grid input to hippocampus
        #g2p = self.g2p(g)

        ########################################
        #g2p_all = tf.tile(tf.expand_dims(g2p, axis=2), multiples=[1,1,self.spike_step]) 
        #g2p_all = self.poisson_spike(g2p_all)
        #x2p_all = tf.tile(tf.expand_dims(x2p, axis=2), multiples=[1,1,self.spike_step]) 
        #x2p_all = self.poisson_spike(x2p_all)
        #x2p_all = self.gumbel_sigmoid(x2p_all)
        #g_spike = tf.tile(tf.expand_dims(g, axis=2), multiples=[1,1,self.spike_step]) 
        #g_spike = self.poisson_spike(g_spike*10)
        #g_spike = self.gumbel_sigmoid(g_spike*self.g_spike)
        ############      DG     ###################
        v = self.v_dg
        v2 = self.v_dg2
        #o_sum = 0
        #v_dg_sum = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        dg_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        dg_all2 = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        for i in range(self.spike_step):
            o, v = self.dg_spiking((g_spike[:,:,i], v))
            #o2, v2 = self.dg_spiking2((x2p_all[:,:,i], v2))
            o2, v2 = self.dg_spiking2((x2dg_all[:,:,i], v2))
            #o, v = self.dg_spiking((tf.concat([g_spike, x2p_all], axis=1)[:,:,i], v))
            #o, v = self.dg_spiking((g_spike[:,:,i], v))
            #v_dg_sum = v_dg_sum.write(i, tf.where(o == 1, tf.ones_like(v), tf.zeros_like(v)))
            #o_sum += o / self.spike_step
            dg_all = dg_all.write(i, o)
            dg_all2 = dg_all2.write(i, o2)
        #print("VVV",v)
        #v_dg_sum = v_dg_sum.stack() 
        #v_dg_sum = tf.transpose(v_dg_sum, perm=[1, 2, 0])
        self.v_dg.assign(v)
        dg_all = dg_all.stack() 
        dg_all = tf.transpose(dg_all, perm=[1, 2, 0]) 
        self.v_dg2.assign(v2)
        dg_all2 = dg_all2.stack() 
        dg_all2 = tf.transpose(dg_all2, perm=[1, 2, 0]) 

        #############  CA3 ########################
        v = self.v_ca3
        #o_sum = 0
        #v_sum = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        #tf.print("GGGGG",g_spike, summarize=-1)
        #x_2p_all = tf.zeros((x.shape[0], self.par.p_size, self.spike_step)) 
        ca3_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        #print("g_spike[:,:,i]",g_spike[:,:,i], ca3_prev)
        #tf.print("CCC",ca3_prev,summarize=-1)
        for i in range(self.spike_step):
            ca3_input = tf.concat([
                #g_spike[:,:,i],
                dg_all[:,:,i],
                #dg_all2[:,:,i],
                #x2p_all[:,:,i],
                x2dg_all [:,:,i],
                ca3_prev[:,:,i]
            ], axis=1)
            o, v = self.ca3_spiking((ca3_input, v))
            ca3_all = ca3_all.write(i, o)
            #ca3_prev = o
        #print("VVV",v)
        #v_sum = v_sum.stack() 
        #v_sum = tf.transpose(v_sum, perm=[1, 2, 0])
        self.v_ca3.assign(v)
        ca3_all = ca3_all.stack() 
        ca3_all = tf.transpose(ca3_all, perm=[1, 2, 0])

        ###########  CA1  ###################################
        v = self.v_infer_p
        p_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        for i in range(self.spike_step):
            #o, v = self.infer_p_spiking((g2p_all[:,:,i], v))
            #o, v = self.infer_p_spiking((ca3_all[:,:,i], v))
            #o, v = self.infer_p_spiking(((ca3_all * x2p_all)[:,:,i], v))
            #o, v = self.infer_p_spiking((tf.concat([g2p_all, x2p_all], axis=1)[:,:,i], v))
            o, v = self.infer_p_spiking((tf.concat([ca3_all, x2p_all], axis=1)[:,:,i], v))
            #o, v = self.infer_p_spiking((tf.concat([g2p_all, x_spike], axis=1)[:,:,i], v))
            #o, v = self.infer_p_spiking((tf.concat([g2p_all, x2lec_all], axis=1)[:,:,i], v))
            #v_sum.append(tf.where(o == 1, tf.ones_like(v)*self.par['thr'], v))
            #o, v = self.infer_p_spiking((x_two_hot_spike[:,:,i], v))
            #o, v = self.infer_p_spiking((tf.concat([g_spike, x_spike], axis=1)[:,:,i], v))
            #o, v = self.infer_p_spiking((tf.concat([g_spike, tf.stop_gradient(x_two_hot_spike)], axis=1)[:,:,i], v))
            #o, v = self.infer_p_spiking((tf.concat([g_spike, x2p_all], axis=1)[:,:,i], v))
            #p_all = p_all.write(i, o*self.m_inf_p)
            p_all = p_all.write(i, o)
        #print("FFFF",v)
        self.v_infer_p.assign(v)
        p_all = p_all.stack()  # shape: (spike_step, batch_size, p_size)
        p_spike = tf.transpose(p_all, perm=[1, 2, 0])
        p = tf.reduce_mean(p_spike, axis=2)

        return p, p_spike, dg_all, ca3_all, dg_all2, ca3_all

    @model_utils.define_scope
    def x2p(self, x, x_t, x_two_hot, d):
        if self.par.two_hot:
            # if using two hot encoding of sensory stimuli
            x_comp = x_two_hot
        else:
            # otherwise compress one-hot encoding
            x_comp = x_two_hot #self.MLP_c(x)

        ######## spiking #####################
        #x_spike = tf.tile(tf.expand_dims(x, axis=2), multiples=[1,1,self.spike_step]) 
        x_spike = tf.tile(tf.expand_dims(x_t[0], axis=2), multiples=[1,1,self.spike_step]) 
        x_spike = self.poisson_spike(x_spike)
        #v = tf.zeros((x.shape[0], self.par['p_size']*self.par['k'])) 
        v = self.v_x2p
        v2 = self.v_x2dg
        #o_sum = 0
        #x_2p_all = tf.zeros((x.shape[0], self.par.p_size, self.spike_step)) 
        x_2p_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        x2dg_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        #v_x_2p_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        for i in range(self.spike_step):
            o, v = self.x2p_spiking((x_spike[:,:,i], v))
            o2, v2 = self.x2dg_spiking((x_spike[:,:,i], v2))
            #v_x_2p_all = v_x_2p_all.write(i, tf.where(o == 1, tf.ones_like(v), tf.zeros_like(v)))
            #print("VVVV2",v)
            #o_sum += o / self.spike_step
            x_2p_all = x_2p_all.write(i, o)
            x2dg_all = x2dg_all.write(i, o2)
        self.v_x2p.assign(v)
        x_2p_all = x_2p_all.stack()  # shape: (spike_step, batch_size, p_size)
        x_2p_all = tf.transpose(x_2p_all, perm=[1, 2, 0]) 
        self.v_x2dg.assign(v2)
        x2dg_all = x2dg_all.stack()  # shape: (spike_step, batch_size, p_size)
        x2dg_all = tf.transpose(x2dg_all, perm=[1, 2, 0]) 
        #v_x_2p_all = v_x_2p_all.stack()  # shape: (spike_step, batch_size, p_size)
        #v_x_2p_all = tf.transpose(v_x_2p_all, perm=[1, 2, 0]) 
        """o_sum= tf.reshape(o_sum, (self.par['p_size'],self.par['k']))
        x_2p_all= tf.reshape(x_2p_all, (self.par['p_size'],self.par['k'], self.spike_step))
        random_indices= tf.random.uniform(shape=(self.par['p_size'],), minval=0, maxval=self.par['k'], dtype=tf.int32)
        batch_indices = tf.range(self.par['p_size'], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, random_indices], axis=1)  # shape: (15, 2)
        o_sum = tf.gather_nd(o_sum, gather_indices)
        o_sum = tf.reshape(o_sum, (1, self.par['p_size'],))
        x_2p_all = tf.gather_nd(x_2p_all, gather_indices)
        x_2p_all = tf.reshape(x_2p_all, (1, self.par['p_size'], self.spike_step,))"""
        #if self.par['stdp']:
        #    x_2p_all = self.lif_stdp_output(x_2p_all, self.stdp_W_p2g)
        #x_2p_all = tf.expand_dims(x_2p_all, axis=0)
        #tf.print("x_2p_all",x_2p_all,summarize=-1)
        x2p = tf.reduce_mean(x_2p_all, axis=2)#o_sum

        # temporally filter
        x_ = self.x2x_(x_comp, x_t, d)
        #tf.print("XXXXXXXXXX", x_[0].shape, x.shape)
        # normalise
        #x_normed = self.f_n(x_)
        # tile to make same size as hippocampus
        #x_2p = self.x_2p(x_normed)

        return x2p, x_, x_2p_all, x_spike, x2dg_all

    @model_utils.define_scope
    def x2x_(self, x, x_, d):
        """
        Temporally filter data in different frequency bands
        :param x: input (compressed or otherwise
        :param x_: previous filtered values
        :param d:
        :return: new filtered values
        """

        x_s = []
        for f in range(self.par.n_freq):
            # get filtering parameter for each frequency
            # inverse sigmoid as initial parameters
            a = tf.sigmoid(self.gamma[f])

            # filter
            filtered = a * x_[f] + x * (1 - a)
            if self.par.smooth_only_on_movement:
                # only filter if actually moved
                stay_still = tf.reduce_sum(d, axis=1, keepdims=True)
                filtered = filtered * stay_still + (1 - stay_still) * x_[f]
            x_s.append(filtered)

        return x_s
    
    @model_utils.define_scope
    def x2g(self, x, x_t, x_two_hot, d):
        #x_spike = tf.tile(tf.expand_dims(x, axis=2), multiples=[1,1,self.spike_step]) 
        x_spike = tf.tile(tf.expand_dims(x_t[0], axis=2), multiples=[1,1,self.spike_step]) 
        x_spike = self.poisson_spike(x_spike)
        v = self.v_x2lec
        #o_sum = 0
        #x_2p_all = tf.zeros((x.shape[0], self.par.p_size, self.spike_step)) 
        x_2lec_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        #v_x_2p_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        for i in range(self.spike_step):
            o, v = self.x2lec_spiking((x_spike[:,:,i], v))
            #v_x_2p_all = v_x_2p_all.write(i, tf.where(o == 1, tf.ones_like(v), tf.zeros_like(v)))
            #print("VVVV2",v)
            #o_sum += o / self.spike_step
            #x_2lec_all = x_2lec_all.write(i, o*self.m_x2g)
            x_2lec_all = x_2lec_all.write(i, o)
        self.v_x2lec.assign(v)
        x_2lec_all = x_2lec_all.stack()  # shape: (spike_step, batch_size, p_size)
        x_2lec_all = tf.transpose(x_2lec_all, perm=[1, 2, 0]) 
        #v_x_2p_all = v_x_2p_all.stack()  # shape: (spike_step, batch_size, p_size)
        #v_x_2p_all = tf.transpose(v_x_2p_all, perm=[1, 2, 0]) 
        """o_sum= tf.reshape(o_sum, (self.par['p_size'],self.par['k']))
        x_2p_all= tf.reshape(x_2p_all, (self.par['p_size'],self.par['k'], self.spike_step))
        random_indices= tf.random.uniform(shape=(self.par['p_size'],), minval=0, maxval=self.par['k'], dtype=tf.int32)
        batch_indices = tf.range(self.par['p_size'], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, random_indices], axis=1)  # shape: (15, 2)
        o_sum = tf.gather_nd(o_sum, gather_indices)
        o_sum = tf.reshape(o_sum, (1, self.par['p_size'],))
        x_2p_all = tf.gather_nd(x_2p_all, gather_indices)
        x_2p_all = tf.reshape(x_2p_all, (1, self.par['p_size'], self.spike_step,))"""
        #x_2lec = tf.reduce_mean(x_2lec_all, axis=2)#o_sum

        return x_2lec_all#, v_x_2p_all

    @model_utils.define_scope
    def gen_g(self, g, t_mat, seq_pos, input_d, g_spike):

        seq_pos_ = tf.expand_dims(seq_pos, axis=1)

        # generative prior on grids if first step in environment, else transition
        mu_gen, g2g_spike = self.g2g(g, t_mat, input_d, g_spike)

        mu_prior = self.g_prior()

        return tf.where(seq_pos_ > 0, mu_gen, mu_prior), g2g_spike

    @model_utils.define_scope
    def g2g(self, g, t_mat, d, g_spike, name=''):

        # transition update
        """update = self.get_g2g_update(g, t_mat)
        # add on update to current representation
        mu = update + g
        mu = self.activation(mu, 'g')"""
        #####################################
        #g = g[..., tf.newaxis] #* dt  # shape = [B, D, 1]
        #g = tf.tile(g, [1, 1, self.spike_step])  
        #g_spike = self.poisson_spike(g)
        mu_all2 = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        #mu_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
        v2 = self.v_g2g
        step = 0
        for i in range(self.spike_step):
            #o, v2 = self.get_g2g_update2(g_spike[:,:,i]*self.g2g_spike, t_mat, v2)
            #o, v2 = self.get_g2g_update2(g_spike[:,:,i]*self.g2g_spike, t_mat, v2, step)
            o, v2 = self.get_g2g_update2(g_spike[:,:,i], t_mat, v2, step)
            step += 1
            mu_all2 = mu_all2.write(i,o)
            #mu_all = mu_all.write(i,o)
            #mu_all2 = mu_all2.write(i,o*self.g2g_spike)
        mu_all2 = mu_all2.stack()  # shape: (spike_step, batch_size, p_size)
        g2g_spike = tf.transpose(mu_all2, perm=[1, 2, 0])
        #mu_all = mu_all.stack()  # shape: (spike_step, batch_size, p_size)
        #g2g_spike_ = tf.transpose(mu_all, perm=[1, 2, 0])
        mu = tf.reduce_mean(g2g_spike, axis=2)#*self.g2_spike
        self.v_g2g.assign(v2)
        ########################################
        """ds = tf.unstack(d, axis=0)
        g_d = tf.concat([tf.stop_gradient(g),d], axis=1)
        if len(tf.shape(g_d))==2:
            g_d = tf.tile(tf.expand_dims(g_d, axis=2), multiples=[1,1,self.spike_step])
        v = tf.zeros((g_d.shape[0], g.shape[1])) 
        #mu = g#tf.zeros((g.shape[0], g.shape[1]))
        
        for i in range(self.spike_step):
            mu, _ = self.g2g_mu_spike((g_d[:,:,i], v))
            #print("MMMMMMM",mut)
        #    mu += mut
        #print("MMMMMMMMMMMM",mu)"""

        return mu, g2g_spike
    
    @tf.custom_gradient
    def spike_function(x):
        """Surrogate gradient spike function"""
        def grad(dy):
            sigma = 2.0
            # 三角形近似の surrogate 勾配
            surrogate_grad = tf.maximum(0., 1 - tf.abs(x) / sigma)
            return dy * surrogate_grad
        out = tf.cast(x > 0.0, tf.float32)
        return out, grad

    @model_utils.define_scope
    def get_g2g_update2(self, g_p, t_mat, v_prev, step):

        # 1. 入力を重みにかける
        input_current = tf.squeeze(tf.matmul(t_mat, tf.expand_dims(g_p, axis=2)), axis=2)

        # 2. 膜電位の更新（リークあり）
        alpha = tf.exp(-1.0 / self.par['tau'])
        v_new = alpha * v_prev + input_current - self.par['theta_amp'] * (tf.sin(2.0 * np.pi * step /self.spike_step)+1)
        v_new = tf.clip_by_value(v_new, clip_value_min=self.par['v_min'], clip_value_max=self.par['v_max'])

        # 3. スパイク発火
        spike = spike_function(v_new - self.par['thr'])

        # 4. リセット（発火したニューロンだけ 0 付近に戻す）
        v_reset = tf.where(spike > 0.0, self.par['v_reset'], v_new)

        return spike, v_reset

    @model_utils.define_scope
    def g_prior(self):
        """
        Gives prior distribution for grid cells
        :return:
        """

        mu = self.g_init if self.g_init is not None else tf.tile(self.g_prior_mu, [self.batch_size, 1])
        #logsig = tf.tile(self.g_prior_logsig, [self.batch_size, 1]) + self.par.logsig_offset  # JW: diff

        #sigma = tf.exp(logsig)

        return mu#, sigma

    @model_utils.define_scope
    def get_transition(self, d):
        # get transition matrix based on relationship / action
        t_vec = self.t_vec(d)
        # turn vector into matrix
        trans_all = tf.reshape(t_vec, [self.batch_size, self.par.g_size, self.par.g_size])
        # apply mask - i.e. if hierarchically or only transition within frequency
        return trans_all * self.mask_g

    @model_utils.define_scope
    def get_g2g_update(self, g_p, t_mat):

        # multiply current entorhinal representation by transition matrix
        update = tf.squeeze(tf.matmul(t_mat, tf.expand_dims(g_p, axis=2)))

        return update

    @tf.custom_gradient
    def poisson_spike(self, p_spike):
        random_values = tf.random.uniform(tf.shape(p_spike))
        spikes = tf.cast(random_values < p_spike, tf.float32)

        def grad(dy):
            return dy

        return spikes, grad

    @model_utils.define_scope
    def f_x(self, p_spike, name):

        """ps = tf.split(value=p, num_or_size_splits=self.par.n_place_all, axis=1)

        # same as W_tile^T
        x_s = tf.reduce_sum(
            tf.reshape(ps[self.par.prediction_freq], (self.batch_size, self.par.n_phases_all[
                self.par.prediction_freq], self.par.s_size_comp)), axis=1)

        x_logits_ = self.w_x * x_s + self.b_x"""
        #x_logits = self.MLP_c_star(x_logits_)
        #x = tf.nn.softmax(x_logits)
        # decompress sensory
        ############### spiking #############################
        """x_logits_ = tf.nn.softmax(x_logits_)
        if len(tf.shape(x_logits_))==2:
            x_logits_ = tf.tile(tf.expand_dims(x_logits_, axis=1), multiples=[1,self.spike_step,1])
            x_logits = self.MLP_c_star_spiking(x_logits_)
            x_logits = tf.transpose(x_logits, perm=[0, 2, 1])
            x_logits = tf.reduce_mean(x_logits, axis=1)
            x = tf.nn.softmax(x_logits)"""
        
        ################ spiking2 ###################################
        #if len(tf.shape(p))==2:
            #p = tf.tile(tf.expand_dims(x_logits_, axis=2), multiples=[1,1,self.spike_step])
            #p = tf.tile(tf.expand_dims(p, axis=2), multiples=[1,1,self.spike_step])
            #x_logits_ = self.poisson_spike(x_logits_)
            #p_spike = self.poisson_spike(p)
            #p = self.gumbel_sigmoid(p*self.fx_spike)
        #v = tf.zeros((self.par.k, x_logits_.shape[0], self.par.s_size)) 
        if name == "p":
            #p = p_spike
            p2g_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
            v = self.v_p2g_gen 
            for i in range(self.spike_step):
                o, v = self.p2g_gen_spiking((p_spike[:,:,i], v))
                #o, v = self.p2g_gen_spiking((tf.concat([p_spike, x2g_spike, x2rsc_all], axis=1)[:,:,i], v))
                #o, v = self.p2g_gen_spiking((tf.concat([p_spike, x_spike], axis=1)[:,:,i], v))
                #v_sum = v_sum.write(i, tf.where(o == 1, tf.ones_like(v), tf.zeros_like(v)))
                #o_sum += o / self.spike_step
                p2g_all = p2g_all.write(i,o)
            self.v_p2g_gen.assign(v)
            p2g_all = p2g_all.stack()  # shape: (spike_step, batch_size, p_size)
            p2g_all = tf.transpose(p2g_all, perm=[1, 2, 0])

            """g2x_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
            v = self.v_g2x_gen 
            for i in range(self.spike_step):
                o, v = self.g2x_gen_spiking((p2g_all[:,:,i], v))
                g2x_all = g2x_all.write(i,o)
            self.v_g2x_gen.assign(v)
            g2x_all = g2x_all.stack()  # shape: (spike_step, batch_size, p_size)
            g2x_all = tf.transpose(g2x_all, perm=[1, 2, 0])"""
        if name == "p_g":
            v = self.v_p_g_gen
            p2g_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
            for i in range(self.spike_step):
                o, v = self.p2g_gen_spiking((p_spike[:,:,i], v))
                p2g_all = p2g_all.write(i,o)
            self.v_p_g_gen.assign(v)
            p2g_all = p2g_all.stack()  # shape: (spike_step, batch_size, p_size)
            p2g_all = tf.transpose(p2g_all, perm=[1, 2, 0])
        if name == "p_gt":
            p2g_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
            v = self.v_p2gt_gen 
            for i in range(self.spike_step):
                o, v = self.p2g_gen_spiking((p_spike[:,:,i], v))
                #o, v = self.p2g_gen_spiking((tf.concat([p_spike, x2g_spike, x2rsc_all], axis=1)[:,:,i], v))
                #o, v = self.p2g_gen_spiking((tf.concat([p_spike, x_spike], axis=1)[:,:,i], v))
                p2g_all = p2g_all.write(i,o)
            self.v_p2gt_gen.assign(v)
            p2g_all = p2g_all.stack()  # shape: (spike_step, batch_size, p_size)
            p2g_all = tf.transpose(p2g_all, perm=[1, 2, 0])

            """g2x_all = tf.TensorArray(dtype=tf.float32, size=self.spike_step)
            v = self.v_gt2x_gen 
            for i in range(self.spike_step):
                o, v = self.g2x_gen_spiking((p2g_all[:,:,i], v))
                g2x_all = g2x_all.write(i,o)
            self.v_gt2x_gen.assign(v)
            g2x_all = g2x_all.stack()  # shape: (spike_step, batch_size, p_size)
            g2x_all = tf.transpose(g2x_all, perm=[1, 2, 0])"""

        o_sum = 0
        
        if name == "p":
            v = self.v_fx 
            for i in range(self.spike_step):
                #o, v = self.MLP_c_star_spiking2((x_logits_[:,:,i], v))
                #o, v = self.MLP_c_star_spiking2((p[:,:,i], v))
                o, v = self.MLP_c_star_spiking2((p2g_all[:,:,i], v))
                #o, v = self.MLP_c_star_spiking2((g2x_all[:,:,i], v))
                #v_ = tf.where(o > 0, tf.ones_like(v)*2.0, v)
                o_sum += o / self.spike_step
            self.v_fx.assign(v)
        if name == "p_g":
            v = self.v_fx_pg
            for i in range(self.spike_step):
                o, v = self.MLP_c_star_spiking2((g2x_all[:,:,i], v))
                o_sum += o / self.spike_step
            self.v_fx_pg.assign(v)
        if name == "p_gt":
            v = self.v_fx_pgt
            for i in range(self.spike_step):
                #o, v = self.MLP_c_star_spiking2((x_logits_[:,:,i], v))
                #o, v = self.MLP_c_star_spiking2((p[:,:,i], v))
                o, v = self.MLP_c_star_spiking2((p2g_all[:,:,i], v))
                #v_ = tf.where(o > 0, tf.ones_like(v)*2.0, v)
                o_sum += o / self.spike_step
            self.v_fx_pgt.assign(v)
        #tf.print("OOOOOOf",o_sum,summarize=-1)
        """o_sum= tf.reshape(o_sum, (self.par.s_size,self.par.k))
        v = tf.reshape(v, (self.par.s_size,self.par.k))
        #tf.print("OOOOOO",o_sum,summarize=-1)
        random_indices= tf.random.uniform(shape=(self.par.s_size,), minval=0, maxval=self.par.k, dtype=tf.int32)
        #print("random_indices", random_indices)
        batch_indices = tf.range(self.par.s_size, dtype=tf.int32)
        #print("batch_indices",batch_indices)
        gather_indices = tf.stack([batch_indices, random_indices], axis=1)  # shape: (15, 2)
        #tf.print("gather_indices",  gather_indices,summarize=-1)
        o_sum = tf.gather_nd(o_sum, gather_indices)
        o_sum = tf.reshape(o_sum, (1, self.par.s_size,))
        v = tf.gather_nd(v, gather_indices)
        v = tf.reshape(v, (1, self.par.s_size,))"""
        #tf.print("OOOOOOOOOOs",o_sum,summarize=-1)
        #tf.print("random_index",random_index, summarize=-1)
        #o_sum = tf.reshape(o_sum[random_index], (1, 15))
        x_logits = o_sum#spike_Bernoulli
        x = o_sum#spike_Bernoulli
        #######################################################3
        
        #tf.print("XXXXXXXXXXxx",x,summarize=-1)

        return x, x_logits, p2g_all

    
    @tf.custom_gradient
    def surrogate_spike(self, V):
        threshold = self.par['thr'] #0.05
        spike = tf.cast(V > threshold, tf.float32)
        dampening_factor = 1.0 #30a
        def grad(dy):
            sigma = 2.0
            #grad_v = sigma * tf.exp(-sigma * tf.abs(V - threshold))
            grad_v = dampening_factor * tf.maximum(0.,1 - tf.abs(V - threshold)/sigma)
            return dy * grad_v
        
        return spike, grad
    
    def lif_stdp_output(self, pre_spikes, W, v_reset=0.0):
        N_out, N_in = self.stdp_W.shape
        T = pre_spikes.shape[-1]

        V = tf.zeros([N_out], dtype=tf.float32)
        post_spikes = []

        for t in range(T):
            input_t = pre_spikes[0,:, t]
            I_t = tf.linalg.matvec(self.stdp_W, input_t)
            V = tf.exp(-1/self.par.tau) * V + I_t
            #V = tf.clip_by_value(V, clip_value_min=self.par['v_min'], clip_value_max=self.par['v_max'])

            spikes_t = self.surrogate_spike(V) 
            V = tf.where(spikes_t > 0.0, self.par['v_reset'], V)

            post_spikes.append(spikes_t)

        return tf.stack(post_spikes, axis=1)
    
    
    @model_utils.define_scope
    def stdp_update2(self, pre_spikes, post_spikes, W, tau_plus=0.5, tau_minus=0.5, A_plus=0.0005, A_minus=-0.0005, eta_decay=0.0, w_clip=(-0.2, 0.2)):

        _, N_in, T = pre_spikes.shape
        _, N_out, _ = post_spikes.shape

        # Trace initialization
        pre_trace = tf.zeros((N_in,), dtype=tf.float32)
        post_trace = tf.zeros((N_out,), dtype=tf.float32)

        for t in range(T):
            pre_t = pre_spikes[0, :, t]    # (N_in,)
            post_t = post_spikes[0, :, t]  # (N_out,)

            # Update traces (exponential decay + current spike)
            pre_trace = pre_trace * tf.exp(-1.0 / tau_plus) + pre_t
            post_trace = post_trace * tf.exp(-1.0 / tau_minus) + post_t

            # Long-term potentiation (LTP): post fires, look at past pre trace
            dw_ltp = A_plus * tf.tensordot(post_t, pre_trace, axes=0)  # (N_out, N_in)

            # Long-term depression (LTD): pre fires, look at past post trace
            dw_ltd = A_minus * tf.tensordot(post_trace, pre_t, axes=0)  # (N_out, N_in)

            # Apply both updates
            W.assign_add(dw_ltp + dw_ltd)

            # Optional: weight decay (slow drift toward zero)
            #if eta_decay > 0:
            #    W.assign(W * (1.0 - eta_decay))

        # Clip weights to prevent runaway
        W.assign(tf.clip_by_value(W, w_clip[0], w_clip[1]))


    @model_utils.define_scope
    def init_input(self, inputs, new_mems=None):
        """
        Set model member variables from inputs and prepare memory and variable dictionaries
        """
        variable_dict = self.init_vars()
        # Return dicts
        return variable_dict

    @model_utils.define_scope
    def init_vars(self, seq_len=None):
        """
        Collecting variables for losses, accuracies and saving. Start with all fields that can possibly be collected.
        Then when generating output in tensorarray_2_list, only stack those fields that were actually written to.
        Tensorflow annoying any wont deal with list appends with tf.range, so using TensorArray instead        
        """
        # Total number of variables collected: if not provided, default to the length of the backprop sequence
        seq_len = self.par.seq_len if seq_len is None else seq_len

        # Create dictionary with all possible data for saving
        vars_dict = model_utils.DotDict(
            {'g': {'g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g'),
                    'g_2d': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g_2d'),
                   'g_gen': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g_gen'),
                   
                   },
             'p': {'p': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p'),
                   'p_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p_g'),
                   'p_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p_gt'),
                   'p_x': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p_x'),
                   'dg': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_dg'),
                   'ca3': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_ca3')
                   },
             'x_s': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_s'),
             'ca3_prev': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_ca3_prev'),
             'ca1_prev': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_ca1_prev'),
             'p2g_prev': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_p2g_prev'),
             'g_prev': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g_prev'),
             'g2g_spike': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_g2g_spike'),
             'pred': {'x_p': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_p'),
                      'x_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_g'),
                      'x_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_x_gt'),
                      'd': tf.TensorArray(self.precision, size=seq_len - 1, clear_after_read=False, name='ta_d')
                      },
             'logits': {
                 'x_p': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_p'),
                 'x_g': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_g'),
                 'x_gt': tf.TensorArray(self.precision, size=seq_len, clear_after_read=False, name='ta_logit_x_gt'),
                 'd': tf.TensorArray(self.precision, size=seq_len - 1, clear_after_read=False, name='ta_logit_d')
             },
             'weights': {'gamma': 0.0}
             })

        return vars_dict

    @model_utils.define_scope
    def update_vars(self, input_dict, updates, i):
        """
        Select specific fields for writing to an output array, or by default write all used values
        """
        # Create output vars_dict, which will have the requested fields
        output_dict = {}

        # Find all keys in input dict
        all_keys = model_utils.get_all_keys(input_dict)

        # Get updated keys and values by 'transposing' updates input
        update_keys, update_vals = [list(field) for field in zip(*updates)]

        # Run through all keys. Simply copy from input dict, unless new value was specified as update
        for key in all_keys:
            # Get tensorarray from input dict
            input_val = model_utils.nested_get(input_dict, key)
            if key in update_keys:
                # If an update was provided: set update to corresponding value
                model_utils.nested_set(output_dict, key, input_val.write(i, update_vals[update_keys.index(key)]))
            else:
                # If no update was provided: simply copy the original value
                model_utils.nested_set(output_dict, key, input_val)

        # Return output dict
        return model_utils.DotDict(output_dict)

    @model_utils.define_scope
    def precomp_trans(self, dirs, seq_len=None, name=None):
        # If sequence length is not specified: use full sequence length from parameters
        seq_len = self.par.seq_len if seq_len is None else seq_len
        # alternatively could pre-compute all types of actions and then use control flow
        ta_mat = tf.TensorArray(self.precision, size=seq_len, clear_after_read=False,
                                name='t_mat' + ('' if name is None else name))
        ds = tf.unstack(dirs, axis=0)
        for j, d in enumerate(ds):
            # Get transition matrix from action/relation
            new_ta = self.get_transition(d)
            # And write transitions for this iteration to ta_mat
            ta_mat = ta_mat.write(j, new_ta)
        return ta_mat

    @model_utils.define_scope
    def tensorarray_2_list_old(self, variable_dict):
        # likely not the best way to do this...
        vars_dict = model_utils.DotDict({'g': {'g': tf.unstack(variable_dict.g.g.stack(), axis=0, name='g_unstack'),
                                                'g_2d': tf.unstack(variable_dict.g.g_2d.stack(), axis=0, name='g_2d_unstack'),
                                               'g_gen': tf.unstack(variable_dict.g.g_gen.stack(), axis=0,
                                                                   name='g_gen_unstack'),
                                               },
                                         'p': {'p': tf.unstack(variable_dict.p.p.stack(), axis=0, name='p_unstack'),
                                               'p_g': tf.unstack(variable_dict.p.p_g.stack(), axis=0,
                                                                 name='p_g_unstack'),
                                               'p_x': tf.unstack(variable_dict.p.p_x.stack(), axis=0,
                                                                 name='p_x_unstack'),
                                                'dg': tf.unstack(variable_dict.p.dg.stack(), axis=0,
                                                                 name='p_dg_unstack'),
                                                'ca3': tf.unstack(variable_dict.p.ca3.stack(), axis=0,
                                                                 name='p_ca3_unstack')
                                               },
                                         'x_s': tf.unstack(variable_dict.x_s.stack(), axis=0, name='xs_unstack'),
                                         'ca3_prev': tf.unstack(variable_dict.ca3_prev.stack(), axis=0, name='ca3_prev_unstack'),
                                         'ca1_prev': tf.unstack(variable_dict.ca1_prev.stack(), axis=0, name='ca1_prev_unstack'),
                                         'p2g_prev': tf.unstack(variable_dict.p2g_prev.stack(), axis=0, name='p2g_prev_unstack'),
                                         'g_prev': tf.unstack(variable_dict.g_prev.stack(), axis=0, name='g_prev_unstack'),
                                         'g2g_spike': tf.unstack(variable_dict.g2g_spike.stack(), axis=0, name='g2g_spike_unstack'),
                                         'pred': {'x_p': tf.unstack(variable_dict.pred.x_p.stack(), axis=0,
                                                                    name='x_p_unstack'),
                                                  'x_g': tf.unstack(variable_dict.pred.x_g.stack(), axis=0,
                                                                    name='x_g_unstack'),
                                                  'x_gt': tf.unstack(variable_dict.pred.x_gt.stack(), axis=0,
                                                                     name='x_gt_unstack'),
                                                  },
                                         'logits': {
                                             'x_p': tf.unstack(variable_dict.logits.x_p.stack(), axis=0,
                                                               name='x_p_unstack'),
                                             'x_g': tf.unstack(variable_dict.logits.x_g.stack(), axis=0,
                                                               name='x_g_unstack'),
                                             'x_gt': tf.unstack(variable_dict.logits.x_gt.stack(), axis=0,
                                                                name='x_gt_unstack'),
                                         },
                                         'weights': {'gamma': self.gamma}
                                         })

        # Add action predictions, if they exist
        if 'd' in variable_dict.pred:
            # Note ['pred']['d'] instead of .pred.d: DotDict nested assignment doesn't work
            vars_dict['pred']['d'] = tf.unstack(variable_dict.pred.d.stack(), axis=0, name='d_unstack')
            vars_dict['logits']['d'] = tf.unstack(variable_dict.logits.d.stack(), axis=0, name='d_unstack')

        vars_dict.x_s = [tf.unstack(x, axis=0, name='xs_unstack_') for x in vars_dict.x_s]

        return vars_dict

    @model_utils.define_scope
    def tensorarray_2_list(self, variable_dict):
        """
        Select specific fields for writing to an output array, or by default write all used values
        """
        # If no selection of keys to write was provided: simply select all 
        keys_to_write = model_utils.get_all_keys(variable_dict)

        # Create output vars_dict, which will have the requested fields
        vars_dict = {}

        # Then set the values of vars_dict according to fields to write from the input variable dict
        for key in keys_to_write:
            # Retrieve the value of the nested key from input variable dict and stack
            value = model_utils.nested_get(variable_dict, key)
            # Convert value to list if it is a tensorarray
            if isinstance(value, tf.TensorArray):
                # Convert tensorarray to list if it was written to at least once
                value = None if value.element_shape == tf.TensorShape(None) \
                    else tf.unstack(value.stack(), axis=0, name=key[-1] + '_unstack')
            # Set the value of the nested key in the output dict
            model_utils.nested_set(vars_dict, key, value)

        # Return output dict
        return model_utils.DotDict(vars_dict)


@model_utils.define_scope
def compute_losses(model_inputs, data, trainable_variables, par):
    lx_p = 0.0
    lx_g = 0.0
    lx_g_ = 0.0
    lx_gt = 0.0
    lp_x_ = 0.0
    lp = 0.0
    lg = 0.0
    lp_x = 0.0
    lg_reg = 0.0
    lp_reg = 0.0
    l_gp = 0.0

    xs = model_inputs.x
    scalings = model_inputs.scalings
    s_visited = model_inputs.s_visited
    positions = model_inputs.positions

    s_visited_ = tf.unstack(s_visited, axis=1)
    for i in range(par.seq_len):

        if par.world_type in ['loop_laps', 'splitter', 'in_out_bound', 'tank', 'splitter_grieves'] + \
                ['wood2000', 'frank2000', 'grieves2016', 'sun2020', 'nieh2021'] and par.use_reward:
            # are we at a reward state?
            # do we want to increase prediction if at no - or - rewarded state? I.e. x_mult for R and NR in splitters?
            x_mult = tf.where(
                tf.reduce_min(tf.abs(model_inputs.reward_pos - tf.expand_dims(positions[i], axis=1)), axis=1) == 0,
                model_inputs.reward_val, 1.0)

        else:
            x_mult = 1.0

        # losses for each batch
        lx_p_ = 1*model_utils.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_p[i])
        #lx_g_ = model_utils.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_g[i])
        lx_gt_ = 1*model_utils.sparse_softmax_cross_entropy_with_logits(xs[i], data.logits.x_gt[i])
        #lx_p_ = 5*model_utils.squared_error(xs[i], data.logits.x_p[i])
        #tf.print("XXX",xs[i]*1.0, "data.logits.x_p[i]", data.logits.x_p[i],summarize=-1)
        #tf.print("LLL", lx_p_, summarize=-1)
        #lx_g_ = 0*model_utils.squared_error(xs[i]*1.0, data.logits.x_g[i])
        #lx_gt_ = 5*model_utils.squared_error(xs[i], data.logits.x_gt[i])

        lp_ = 1*model_utils.squared_error(data.p.p[i], data.p.p_g[i]) #+ 1*model_utils.squared_error(data.p.ca3[i], tf.reduce_mean(data.ca3_prev[i],axis=2)) + 1*model_utils.squared_error(data.p.p[i], tf.reduce_mean(data.ca1_prev[i],axis=2))
        #lp_x_ = model_utils.squared_error(data.p.p[i], data.p.p_x[i]) if 'lp_x' in par.which_costs else 0
        lg_ = 1*model_utils.squared_error(data.g.g[i], data.g.g_gen[i])
        #lg_ = model_utils.squared_error(data.g_prev[i], data.g2g_spike[i])

        lg_reg_ = 0#*model_utils.squared_error(data.g.g[i], data.g.g_2d[i])+0*tf.reduce_sum(data.g.g[i] ** 2, axis=1)+1*tf.reduce_sum(data.g.g_gen[i] ** 2, axis=1)

        lp_reg_ = 1*tf.reduce_sum(tf.abs(data.p.p_g[i]), axis=1) + 0*tf.reduce_sum(tf.abs(data.p.p[i]), axis=1) + 2*tf.reduce_sum(tf.abs(data.p.dg[i]), axis=1) + 0*tf.reduce_sum(tf.abs(data.p.ca3[i]), axis=1)

        # don't train on any time-steps without when haven't visited that state before.
        s_vis = s_visited_[i]
        batch_vis = tf.reduce_sum(s_vis) + eps
        # normalise for bptt sequence length
        norm = 1.0 / (batch_vis * par.seq_len)

        lx_p += tf.reduce_sum(lx_p_ * s_vis * x_mult) * norm
        lx_g += tf.reduce_sum(lx_g_ * s_vis * x_mult) * norm
        lx_gt += tf.reduce_sum(lx_gt_ * s_vis * x_mult) * norm
        lp += tf.reduce_sum(lp_ * s_vis) * scalings.temp * norm
        lg += tf.reduce_sum(lg_ * s_vis) * scalings.temp * norm
        lp_x += tf.reduce_sum(lp_x_ * s_vis) * scalings.p2g_use * scalings.temp * norm

        lg_reg += tf.reduce_sum(lg_reg_ * s_vis) * par.g_reg_pen * norm #* scalings.g_cell_reg
        lp_reg += tf.reduce_sum(lp_reg_ * s_vis) * par.p_reg_pen * norm #* scalings.p_cell_reg 

    losses = model_utils.DotDict()
    cost_all = 0
    #loss_torus = model_utils.gauss_bonnet_loss(data.g.g_gen, n_x=int(par['g_size']**0.5), n_y=int(par['g_size']**0.5))
    #print("l_gp", l_gp)
    
    if 'lx_gt' in par.which_costs:
        cost_all += lx_gt
        losses.lx_gt = lx_gt
    if 'lx_p' in par.which_costs:
        cost_all += lx_p
        losses.lx_p = lx_p
    if 'lx_g' in par.which_costs:
        cost_all += lx_g
        losses.lx_g = lx_g
    if 'lg' in par.which_costs:
        cost_all += lg
        losses.lg = lg
    if 'lp' in par.which_costs:
        cost_all += lp
        losses.lp = lp
    if 'lp_x' in par.which_costs:
        cost_all += lp_x
        losses.lp_x = lp_x
    if 'lg_reg' in par.which_costs:
        #lg_torus = model_utils.gauss_bonnet_loss(data.g.g, n_x=int(par['g_size']**0.5), n_y=int(par['g_size']**0.5))+model_utils.gauss_bonnet_loss(data.g.g_gen, n_x=int(par['g_size']**0.5), n_y=int(par['g_size']**0.5)) #1*tf.reduce_sum(model_utils.compute_gb(data.g.g_2d)) * par.g_reg_pen
        cost_all += lg_reg #+ lg_torus
        losses.lg_reg = lg_reg #+ lg_torus
    if 'lp_reg' in par.which_costs:
        cost_all += lp_reg
        losses.lp_reg = lp_reg
    if 'l_gp' in par.which_costs:

        cost_all += lp_reg
        losses.l_gp = l_gp
    if 'weight_reg' in par.which_costs:
        losses.weight_reg = tf.add_n(
            [tf.nn.l2_loss(v) for v in trainable_variables if 'bias' not in v.name]) * par.weight_reg_val / par.seq_len
        cost_all += losses.weight_reg

    losses.train_loss = cost_all

    return losses