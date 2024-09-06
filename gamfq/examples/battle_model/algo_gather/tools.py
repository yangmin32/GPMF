import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os


class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'


class Buffer:
    def __init__(self):
        pass

    def push(self, **kwargs):
        raise NotImplementedError


class MetaBuffer(object):
    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):
        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0
        
        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def reset_new(self, start, value):
        self.data[start:] = value


class EpisodesBufferEntry:
    """Entry for episode buffer"""
    def __init__(self):
        self.features = []
        self.features2 = []
        self.features3 = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.terminal = False

    def append(self, feature, feature2, feature3, action, reward, alive, probs=None):
        self.features.append(feature.copy())
        self.features2.append(feature2.copy())
        self.features3.append(feature3.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if probs is not None:
            self.probs.append(probs)
        if not alive:
            self.terminal = True


class EpisodesBuffer(Buffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """
    def __init__(self, use_mean=False):
        super().__init__()
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        feature, feature2, feature3 = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        alives = kwargs['alives']
        ids = kwargs['ids']

        if self.use_mean:
            probs = kwargs['prob']

        buffer = self.buffer
        index = np.random.permutation(len(feature))

        for i in range(len(ids)):
            i = index[i]
            entry = buffer.get(ids[i])
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[ids[i]] = entry

            if self.use_mean:
                entry.append(feature[i], feature2[i], feature3[i], acts[i], rewards[i], alives[i], probs=probs[i])
            else:
                entry.append(feature[i], feature2[i], feature3[i], acts[i], rewards[i], alives[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    def episodes(self):
        """ get episodes """
        return self.buffer.values()


class AgentMemory(object):
    def __init__(self, feat_shape, feat_shape2, feat_shape3, act_n, max_len, use_mean=False):
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.feat2 = MetaBuffer(feat_shape2, max_len)
        self.feat3 = MetaBuffer(feat_shape3, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean

        if self.use_mean:
            self.prob = MetaBuffer((act_n,), max_len)

    def append(self, feat0, feat2, feat3, act, reward, alive, prob=None):
        self.feat0.append(np.array([feat0]))
        self.feat2.append(np.array([feat2]))
        self.feat3.append(np.array([feat3]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=np.bool))

        if self.use_mean:
            self.prob.append(np.array([prob]))

    def pull(self):
        res = {
            'feat0': self.feat0.pull(),
            'feat2': self.feat2.pull(),
            'feat3': self.feat3.pull(),
            'act': self.actions.pull(),
            'rewards': self.rewards.pull(),
            'terminals': self.terminals.pull(),
            'prob': None if not self.use_mean else self.prob.pull()
        }

        return res


class MemoryGroup(object):
    def __init__(self, feat_shape, feat_shape2, feat_shape3, act_n, max_len, batch_size, sub_len, use_mean=False):
        self.agent = dict()
        self.max_len = max_len
        self.batch_size = batch_size
        self.feat_shape = feat_shape
        self.feat_shape2 = feat_shape2
        self.feat_shape3 = feat_shape3
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.act_n = act_n

        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.feat2 = MetaBuffer(feat_shape2, max_len)
        self.feat3 = MetaBuffer(feat_shape3, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        if use_mean:
            self.prob = MetaBuffer((act_n,), max_len)
        self._new_add = 0

    def _flush(self, **kwargs):
        self.feat0.append(kwargs['feat0'])
        self.feat2.append(kwargs['feat2'])
        self.feat3.append(kwargs['feat3'])
        self.actions.append(kwargs['act'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['terminals'])

        if self.use_mean:
            self.prob.append(kwargs['prob'])

        mask = np.where(kwargs['terminals'] == True, False, True)
        mask[-1] = False
        self.masks.append(mask)

    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['ids']):
            if self.agent.get(_id) is None:
                self.agent[_id] = AgentMemory(self.feat_shape, self.feat_shape2, self.feat_shape3, self.act_n, self.sub_len, use_mean=self.use_mean)
            if self.use_mean:
                self.agent[_id].append(feat0=kwargs['state'][0][i], feat2=kwargs['state'][1][i], feat3=kwargs['state'][2][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i], prob=kwargs['prob'][i])
            else:
                self.agent[_id].append(feat0=kwargs['state'][0][i], feat2=kwargs['state'][1][i], feat3 = kwargs['state'][2][i], act=kwargs['acts'][i], reward=kwargs['rewards'][i], alive=kwargs['alives'][i])

    def tight(self):
        ids = list(self.agent.keys())
        np.random.shuffle(ids)
        for ele in ids:
            tmp = self.agent[ele].pull()
            self._new_add += len(tmp['feat0'])
            self._flush(**tmp)
        self.agent = dict()  # clear

    def sample(self):
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        next_idx = (idx + 1) % self.nb_entries

        feature = self.feat0.sample(idx)
        feature_next = self.feat0.sample(next_idx)
        feature2 = self.feat2.sample(idx)
        feature2_next = self.feat2.sample(next_idx)
        feature3 = self.feat3.sample(idx)
        feature3_next = self.feat3.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)

        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
            return feature, feature2, feature3, actions, act_prob, feature_next, feature2_next, feature3_next, act_next_prob, rewards, dones, masks
        else:
            return feature, feature2, feature3, feature_next, feature2_next, feature3_next, dones, rewards, actions, masks

    def get_batch_num(self):
        print('\n[INFO] Length of buffer and new add:', len(self.feat0), self._new_add)
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        return res

    @property
    def nb_entries(self):
        return len(self.feat0)


class SummaryObj:
    """
    Define a summary holder
    """
    def __init__(self, log_dir, log_name, n_group=1):
        self.name_set = set()
        self.gra = tf.Graph()
        self.n_group = n_group

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with self.gra.as_default():
            self.sess = tf.Session(graph=self.gra, config=sess_config)
            self.train_writer = tf.summary.FileWriter(log_dir + "/" + log_name, graph=tf.get_default_graph())
            self.sess.run(tf.global_variables_initializer())

    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """

        with self.gra.as_default():
            for name in name_list:
                if name in self.name_set:
                    raise Exception("You cannot define different operations with same name: `{}`".format(name))
                self.name_set.add(name)
                setattr(self, name, [tf.placeholder(tf.float32, shape=None, name='Agent_{}_{}'.format(i, name))
                                     for i in range(self.n_group)])
                setattr(self, name + "_op", [tf.summary.scalar('Agent_{}_{}_op'.format(i, name), getattr(self, name)[i])
                                             for i in range(self.n_group)])

    def write(self, summary_dict, step):
        """Write summary related to a certain step

        Parameters
        ----------
        summary_dict: dict, summary value dict
        step: int, global step
        """

        assert isinstance(summary_dict, dict)

        for key, value in summary_dict.items():
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):
                for i in range(self.n_group):
                    self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[i], feed_dict={
                        getattr(self, key)[i]: value[i]}), global_step=step)
            else:
                self.train_writer.add_summary(self.sess.run(getattr(self, key + "_op")[0], feed_dict={
                        getattr(self, key)[0]: value}), global_step=step)

class Runner(object):
    def __init__(self, sess, env, handles, map_size, max_steps, models,
                play_handle, pomfq_position = 2, pursuit=False, render_every=None, save_every=None, tau=None, log_name=None, log_dir=None, model_dir=None, train=False):
        """Initialize runner

        Parameters
        ----------
        sess: tf.Session
            session
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int
            render environment interval
        save_every: int
            states the interval of evaluation for self-play update
        models: list
            contains models
        play_handle: method like
            run game
        tau: float
            tau index for self-play update
        log_name: str
            define the name of log dir
        log_dir: str
            donates the directory of logs
        model_dir: str
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train
        self.pursuit = pursuit
        self.pomfq_position = pomfq_position

        if self.train:
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

            summary_items = ['ave_agent_reward', 'total_reward', 'alive', "Sum_Reward", "Kill_Sum"]
            self.summary.register(summary_items)  # summary register
            self.summary_items = summary_items

            self.sess = sess


            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    def run(self, variant_eps, iteration, win_cnt=None):
        info = {'main': None, 'opponent': None}

        # pass
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'alive': 0.}
        info['opponent'] = {'ave_agent_reward': 0., 'total_reward': 0., 'alive': 0.}

        max_nums, nums, agent_r_records, total_rewards = self.play(env=self.env, n_round=iteration, map_size=self.map_size, max_steps=self.max_steps, handles=self.handles,
                    models=self.models, pomfq_position = self.pomfq_position, print_every=50, eps=variant_eps, render=(iteration + 1) % self.render_every if self.render_every > 0 else False, train=self.train)

        for i, tag in enumerate(['main', 'opponent']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['alive'] = nums[i]
            info[tag]['ave_agent_reward'] = agent_r_records[i]

        if self.train:
            print('\n[INFO] {}'.format(info['main']))

            if self.save_every and (iteration + 1) % self.save_every == 0:
                print(Color.INFO.format('[INFO] Saving model ...'))
                self.models[0].save(self.model_dir + '-0', iteration)
                self.models[1].save(self.model_dir + '-1', iteration)
                self.summary.write(info['main'], iteration)
        else:
            if self.pursuit == False:
                print('\n[INFO] {0} \n {1}'.format(info['main'], info['opponent']))
                if info['main']['alive'] > info['opponent']['alive']:
                    win_cnt['main'] += 1
                elif info['main']['alive'] < info['opponent']['alive']:
                    win_cnt['opponent'] += 1
                else:
                    if info['main']['total_reward'] > info['opponent']['total_reward']:
                        win_cnt['main'] += 1
                    elif info['main']['total_reward'] < info['opponent']['total_reward']:
                        win_cnt['opponent'] += 1
                    else:
                        win_cnt['main'] += 1
                        win_cnt['opponent'] += 1

            else:
                print('\n[INFO] {0} \n {1}'.format(info['main'], info['opponent']))
                if info['main']['alive'] > info['opponent']['alive']:
                    win_cnt['main'] += 1
                elif info['main']['alive'] < info['opponent']['alive']:
                    win_cnt['opponent'] += 1
                else:
                    win_cnt['main'] += 1
                    win_cnt['opponent'] += 1

        return total_rewards,nums

