"""Battle
"""

import argparse
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import magent
import csv
from examples.battle_model.algo_gather import spawn_ai
from examples.battle_model.algo_gather import tools
from examples.battle_model.senario_gather import battle
from examples.battle_model.senario_gather import battle2
from examples.battle_model.senario_gather import battle2_change
from examples.battle_model.senario_gather import battle3
from examples.battle_model.senario_gather import battle3_change

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'pomfq','gamfq','gamfq1'}, help='choose an algorithm from the preset', required=True)
    parser.add_argument('--oppo', type=str, choices={'ac', 'mfac', 'mfq', 'il', 'pomfq','gamfq''gamfq1'}, help='indicate the opponent model')
    parser.add_argument('--n_round', type=int, default=500, help='set the trainning round')
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    parser.add_argument('--map_size', type=int, default=28, help='set the size of map')  # then the amount of agents is 50
    parser.add_argument('--max_steps', type=int, default=500, help='set the max steps')
    parser.add_argument('--pomfq_position', type=int, default=2, help='set the position of pomfq if no pomfq - position is 2')

    parser.add_argument('--idx', nargs='*', required=True)

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('gather', map_size=args.map_size)
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    handles = env.get_handles()
    player_handles = handles[1:]
    pomfq_position = args.pomfq_position

    with open('storepoints_multigather.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Game", "Reward 1", "Reward 2"))
    
    with open('numberofwins_multigather.csv', 'w+') as myfile:
        myfile.write('{0},{1},{2}\n'.format("Game", "nw1", "nw2"))
    
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True

    main_model_dir = os.path.join(BASE_DIR, 'data/gather/models/{}-0'.format(args.algo))
    oppo_model_dir = os.path.join(BASE_DIR, 'data/gather/models/{}-1'.format(args.oppo))

    sess = tf.Session(config=tf_config)
    models = [spawn_ai(args.algo, sess, env, player_handles[0], args.algo + '-me', args.max_steps), spawn_ai(args.oppo, sess, env, player_handles[1], args.oppo + '-opponent', args.max_steps)]
    sess.run(tf.global_variables_initializer())

    models[0].load(main_model_dir, step=args.idx[0])
    models[1].load(oppo_model_dir, step=args.idx[1])

    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, battle3, pomfq_position, render_every=0)
    win_cnt = {'main': 0, 'opponent': 0}

    for k in range(0, args.n_round):
        total_rewards , nums= runner.run(0.0, k, win_cnt=win_cnt)
        with open('gamfq_mfac_storepoints_multigather.csv', 'a') as myfile:
            myfile.write('{0},{1},{2},{3},{4}\n'.format(k, total_rewards[0], total_rewards[1],nums[0],nums[1]))
    
        with open('numberofwins_multigather.csv', 'a') as myfile:
            myfile.write('{0},{1},{2}\n'.format(k, win_cnt['main'], win_cnt['opponent']))
    
    print('\n[*] >>> WIN_RATE: [{0}] {1} / [{2}] {3}'.format(args.algo, win_cnt['main'] / args.n_round, args.oppo, win_cnt['opponent'] / args.n_round))
