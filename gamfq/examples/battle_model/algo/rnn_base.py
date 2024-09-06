import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from magent.gridworld import GridWorld


class ValueNet:
    def __init__(self, sess, env, handle, name, update_every=5, use_mf=False, learning_rate=1e-4, tau=0.005, gamma=0.95, maxelements=20):
        # assert isinstance(env, GridWorld)
        self.env = env
        self.name = name
        self._saver = None
        self.sess = sess

        self.handle = handle
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        self.feature_space = env.get_feature_space(handle)
        self.num_actions = env.get_action_space(handle)[0]
        count = maxelements * 5
        self.new_space = (count,)
        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        self.temperature = 0.1

        self.lr= learning_rate
        self.tau = tau
        self.gamma = gamma
        self.batch_size = 32
        self.unroll_step = 2
        self.skip_error = 0 
        self.pad_before_len = self.unroll_step - 1
        self.agent_states = {}
        with tf.variable_scope(name or "ValueNet"):
            self.name_scope = tf.get_variable_scope().name
            #self.obs_input = tf.placeholder(tf.float32, (None,) + self.view_space, name="Obs-Input")
            self.new_input = tf.placeholder(tf.float32, (None,) + self.new_space, name="New-Input")
            self.feat_input = tf.placeholder(tf.float32, (None,) + self.feature_space, name="Feat-Input")
            self.mask = tf.placeholder(tf.float32, shape=(None,), name='Terminate-Mask')
            self.batch_size_ph   = tf.placeholder(tf.int32, [])
            self.unroll_step_ph = tf.placeholder(tf.int32, [])
            if self.use_mf:
                self.act_prob_input = tf.placeholder(tf.float32, (None, self.num_actions), name="Act-Prob-Input")

            self.act_input = tf.placeholder(tf.int32, (None,), name="Act")
            self.act_one_hot = tf.one_hot(self.act_input, depth=self.num_actions, on_value=1.0, off_value=0.0)

            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name
                self.e_q, self.state_in, self.rnn_state = self._construct_net(active_func=tf.nn.relu)
                self.predict = tf.nn.softmax(self.e_q / self.temperature)
                self.e_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)

            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_q, self.target_state_in, self.target_rnn_state = self._construct_net(active_func=tf.nn.relu)
                self.t_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            with tf.variable_scope("Update"):
                self.update_op = [tf.assign(self.t_variables[i],
                                            self.tau * self.e_variables[i] + (1. - self.tau) * self.t_variables[i])
                                    for i in range(len(self.t_variables))]

            with tf.variable_scope("Optimization"):
                self.target_q_input = tf.placeholder(tf.float32, (None,), name="Q-Input")
                self.e_q_max = tf.reduce_sum(tf.multiply(self.act_one_hot, self.e_q), axis=1)
                self.loss = tf.reduce_sum(tf.square(self.target_q_input - self.e_q_max) * self.mask) / tf.reduce_sum(self.mask)
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct_net(self, active_func=None, reuse=False):

        h_emb = tf.layers.dense(self.feat_input, units=32, activation=active_func,
                                name="Dense-Emb", reuse=reuse)

        h_emb2 = tf.layers.dense(self.new_input, units=32, activation=active_func,
                                name="Dense-Emb2", reuse=reuse)
        concat_layer = tf.concat([h_emb, h_emb2], axis=1)
        k = 2
        if self.use_mf:
            prob_emb = tf.layers.dense(self.act_prob_input, units=64, activation=active_func, name='Prob-Emb')
            h_act_prob = tf.layers.dense(prob_emb, units=32, activation=active_func, name="Dense-Act-Prob")
            concat_layer = tf.concat([concat_layer, h_act_prob], axis=1)
            k = 3
        #dense2 = tf.layers.dense(concat_layer, units=128, activation=active_func, name="Dense2")
        #out = tf.layers.dense(dense2, units=64, activation=active_func, name="Dense-Out")

        
        kernel_num = [32, 32]
        hidden_size = [32]

        state_size = hidden_size[0] * k
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units = state_size)
        rnn_in = tf.reshape(concat_layer, shape=[self.batch_size_ph, self.unroll_step_ph, state_size])
        state_in = rnn_cell.zero_state(self.batch_size_ph, tf.float32)
        rnn, rnn_state = tf.nn.dynamic_rnn(
            cell=rnn_cell, inputs=rnn_in, dtype=tf.float32, initial_state=state_in
        )
        rnn = tf.reshape(rnn, shape=[-1, state_size])
        q = tf.layers.dense(rnn, units=self.num_actions)
        self.state_size = state_size
        return q, state_in, rnn_state

    def _get_agent_states(self, ids):
        """get hidden state of agents"""
        n = len(ids)
        states = np.empty([n, self.state_size])
        default = np.zeros([self.state_size])
        for i in range(n):
            states[i] = self.agent_states.get(ids[i], default)
        return states

    def _set_agent_states(self, ids, states):
        """set hidden state for agents"""
        if len(ids) <= len(self.agent_states) * 0.5:
            self.agent_states = {}
        for i in range(len(ids)):
            self.agent_states[ids[i]] = states[i]



    


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name_scope)

    def calc_target_q(self, **kwargs):
        """Calculate the target Q-value
        kwargs: {'feature', 'feature2', 'prob', 'dones', 'rewards'}
        """
        
        num = len(kwargs['feature'])
        feed_dict = {
            self.feat_input: kwargs['feature'],
            self.new_input: kwargs['feature2'],
            self.batch_size_ph:   self.batch_size,
            self.unroll_step_ph:  self.unroll_step
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        t_q, e_q = self.sess.run([self.t_q, self.e_q], feed_dict=feed_dict)
        act_idx = np.argmax(e_q, axis=1)
        q_values = t_q[np.arange(len(t_q)), act_idx]

        target_q_value = kwargs['rewards'] + (1. - kwargs['dones']) * q_values.reshape(-1) * self.gamma

        return target_q_value

    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)

    def act(self, **kwargs):
        """Act
        kwargs: {'feature', 'feature2', 'prob', 'eps', 'ids'}
        """
        num = len(kwargs['state'][0])
        ids = kwargs['ids']
        states = self._get_agent_states(ids)
        feed_dict = {
            self.feat_input: kwargs['state'][0],
            self.new_input: kwargs['state'][1],
            self.state_in: states,
            self.batch_size_ph: num,
            self.unroll_step_ph: 1
        }


        self.temperature = kwargs['eps']

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input] = kwargs['prob']

        actions, states = self.sess.run([self.predict,self.rnn_state], feed_dict=feed_dict)
        self._set_agent_states(ids, states)
        actions = np.argmax(actions, axis=1).astype(np.int32)
        return actions

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [feature, feature2], 'target_q', 'prob', 'acts'}
        """
        num = len(kwargs['state'][0])
        feed_dict = {
            self.feat_input: kwargs['state'][0],
            self.new_input: kwargs['state'][1],
            self.target_q_input: kwargs['target_q'],
            self.mask: kwargs['masks'],
            self.batch_size_ph:   self.batch_size,
            self.unroll_step_ph:  self.unroll_step
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        feed_dict[self.act_input] = kwargs['acts']
        _, loss, e_q = self.sess.run([self.train_op, self.loss, self.e_q_max], feed_dict=feed_dict)
        return loss, {'Eval-Q': np.round(np.mean(e_q), 6), 'Target-Q': np.round(np.mean(kwargs['target_q']), 6)}
