import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from . import rnn_base
from . import tools

class DQN(rnn_base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, memory_size=2**10, batch_size=64, update_every=5):

        super().__init__(sess, env, handle, name, update_every=update_every)

        self.replay_buffer = tools.MemoryGroup(self.feature_space, self.new_space, self.gather_space, self.num_actions, memory_size, batch_size, sub_len)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            feat, feat2, feat3, feat_next, feat2_next, feat3_next, dones, rewards, actions, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(feature=feat_next, feature2=feat2_next, feature3=feat3_next, rewards=rewards, dones=dones)
            loss, q = super().train(state=[feat, feat2, feat3], target_q=target_q, acts=actions, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "rnndqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "rnndqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))


class MFQ(rnn_base.ValueNet):
    def __init__(self, sess, name, handle, env, sub_len, eps=1.0, update_every=5, memory_size=2**10, batch_size=64):
        super().__init__(sess, env, handle, name, use_mf=True, update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'feat_shape': self.feature_space,
            'feat_shape2': self.new_space,
            'feat_shape3': self.gather_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        self.replay_buffer = tools.MemoryGroup(**config)
        self.update_every = update_every

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_name = self.replay_buffer.get_batch_num()

        for i in range(batch_name):
            feat, feat2, feat3, acts, act_prob, feat_next, feat2_next, feat3_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample()
            target_q = self.calc_target_q(feature=feat_next, feature2=feat2_next, feature3=feat3_next, rewards=rewards, dones=dones, prob=act_prob_next)
            loss, q = super().train(state=[feat, feat2, feat3], target_q=target_q, prob=act_prob, acts=acts, masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "rnnmfq_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope)
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "rnnmfq_{}".format(step))
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))


