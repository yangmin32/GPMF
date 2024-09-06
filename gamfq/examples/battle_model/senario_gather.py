import random
import math
import numpy as np
from scipy.stats import dirichlet
import torch
from examples.battle_model.gt import Gt

def generate_map(env, map_size, food_handle, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)
    center_x, center_y = map_size // 2, map_size // 2

    def add_square(pos, side, gap):
        side = int(side)
        for x in range(center_x - side//2, center_x + side//2 + 1, gap):
            pos.append([x, center_y - side//2])
            pos.append([x, center_y + side//2])
        for y in range(center_y - side//2, center_y + side//2 + 1, gap):
            pos.append([center_x - side//2, y])
            pos.append([center_x + side//2, y])


       # food
    pos = []
    add_square(pos, map_size * 0.65, 10)
    add_square(pos, map_size * 0.6,  10)
    add_square(pos, map_size * 0.55, 10)
    add_square(pos, map_size * 0.5,  4)
    add_square(pos, map_size * 0.45, 3)
    add_square(pos, map_size * 0.4, 1)
    add_square(pos, map_size * 0.3, 1)
    add_square(pos, map_size * 0.3 - 2, 1)
    add_square(pos, map_size * 0.3 - 4, 1)
    add_square(pos, map_size * 0.3 - 6, 1)
    env.add_agents(food_handle, method="custom", pos=pos)

        # legend
    legend = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]

    org = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
    ]

    def draw(base_x, base_y, scale, data):
        w, h = len(data), len(data[0])
        pos = []
        for i in range(w):
            for j in range(h):
                if data[i][j] == 1:
                    start_x = i * scale + base_x
                    start_y = j * scale + base_y
                    for x in range(start_x, start_x + scale):
                        for y in range(start_y, start_y + scale):
                            pos.append([y, x])

        env.add_agents(food_handle, method="custom", pos=pos)

    scale = 1
    w, h = len(legend), len(legend[0])
    offset = -3
    draw(offset + map_size // 2 - w // 2 * scale, map_size // 2 - h // 2 * scale, scale, legend)
    draw(offset + map_size // 2 - w // 2 * scale + len(legend), map_size // 2 - h // 2 * scale, scale, org)


def play(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""  
    #mfq,mfac
    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    nums_per = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list

                
                k = k + 1
        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])
        
        flag = 1
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]
        
        if train:
            models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                new_act_prob[i].append(temp_var[0])

            former_act_prob[i] = np.asarray(new_act_prob[i])
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_nums = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    
    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def play2(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #pomfq
    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    nums_per = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    categories = []
    alphas = {}
    
    for i in range(n_action[0]):
        categories.append(i)
    for i in range(sum(nums)):
        tempe_list = [1] * len(categories)
        alphas[i] = tempe_list
    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list

                
                k = k + 1
        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])
        
        flag = 1
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]
        
        if train:
            models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])

        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                length = len(temp_list)
                tempe_list = alphas[ids[i][j]]
                for k in range(length):
                    tempe_list[temp_list[k]] = tempe_list[temp_list[k]] + 1
                tempe_list = np.asarray(tempe_list)
                alphas[ids[i][j]] = tempe_list
                new_samples = dirichlet.rvs(tempe_list, size = 100, random_state = 1)
                new_mean = np.mean(new_samples, axis = 0)
                new_act_prob[i].append(new_mean)
            
            former_act_prob[i] = np.asarray(new_act_prob[i])
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_num = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def play3(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #gamfq
    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]
    view_buf = [None for _ in range(n_group)]
    adj = [None for _ in range(n_group)]
    feature_buf = [None for _ in range(n_group)]
    prev_hid =[None for _ in range(n_group)]
    hid_size = 64
    batch_size = 1
    detach_gap = 10000
    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    nums_per = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0

    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        listofneighbors_graph0 = {}
        listofneighbors_graph1 = {}          
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            view_buf[i], feature_buf[i] = env.get_observation_whole(handles[i])
            feature_buf[i] = torch.tensor(feature_buf[i])
            feature_buf[i] = feature_buf[i].tolist()
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list

                
                k = k + 1

        def Merge(dict1, dict2):        
            res = {**dict1, **dict2} 
            return res
        listofneighbors = Merge(listofneighbors0,listofneighbors1)

        feature_dict = {}
        for i in range(n_group):
            for j in range(len(identity[i])):
                feature_dict[identity[i][j]] = feature_buf[i][j]

        #######
        graph_dict = {}
        for key in listofneighbors:   
            neighbors = listofneighbors.get(key)   
            # print(neighbors)
            new_feature_graph = []
            a =[]
            if len(neighbors)==0:
                a.append(key)
                graph_dict[key] = a
            else:
                for m in range(len(neighbors)):  
                    neighbor_idx = neighbors[m]
                    # print(new_feature_buf[neighbor_idx])
                    new_feature_graph1 = feature_dict.get(neighbor_idx)  
                    # print(new_feature_graph1)  
                    new_feature_graph.append(new_feature_graph1)  
                new_feature_graph.insert(0,feature_dict.get(key)) 
                num_graph = len(new_feature_graph)
                # print(num_graph,hid_size)
                gt = Gt(num_graph,hid_size)
                prev_hid = torch.zeros(2,num_graph, hid_size) 
                if step_ct == 0:
                    prev_hid = gt.init_hidden(batch_size)  
                new_feature_graph = torch.tensor(new_feature_graph)
                x = [new_feature_graph, prev_hid] 
                adj_id,prev_hid = gt.forward(x)  
                adj=adj_id.tolist() 
                # print(adj[0])   
                neighbors_result=neighbors
                neighbors_result.insert(0,key)
                # print(neighbors_result)
                res_neighbors=[]
                for i in range(len(adj[0])):
                    if adj[0][i] == 1:
                        res_neighbors.append(neighbors_result[i])
                #############################
                if len(res_neighbors)==0:
                    res_neighbors.append(key)
                graph_dict[key] = res_neighbors  
        graph_dict = dict(sorted(graph_dict.items())) 
 
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                # new_list = []
                temp_list = graph_dict.get(identity[i][j]) 
                # # print(temp_list)    
                # for l in range(len(temp_list)):
                #     new_list.append(totalagentids[temp_list[l]])
                if i == 0:
                    listofneighbors_graph0[identity[i][j]] = temp_list
                else:
                    listofneighbors_graph1[identity[i][j]] = temp_list
                # print(listofneighbors_graph0)


        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation_gather1(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation_gather1(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])
        
        flag = 1
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors_graph0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors_graph1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]
        
        if train:
            models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])

        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
                new_act_prob[i].append(temp_var[0])
            
            former_act_prob[i] = np.asarray(new_act_prob[i])
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_num = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def play4(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #gamfq1  Proportional
    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]
    view_buf = [None for _ in range(n_group)]
    adj = [None for _ in range(n_group)]
    feature_buf = [None for _ in range(n_group)]
    prev_hid =[None for _ in range(n_group)]
    hid_size = 64
    batch_size = 1
    detach_gap = 10000
    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    x = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0

    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        listofneighbors_graph0 = {}
        listofneighbors_graph1 = {}             
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            view_buf[i], feature_buf[i] = env.get_observation_whole(handles[i])
            view_buf[i] = torch.tensor(view_buf[i])
            feature_buf[i] = torch.tensor(feature_buf[i])
            feature_buf[i] = feature_buf[i].tolist()
            view_buf[i] = view_buf[i].view(view_buf[i].size(0), -1) 
            # view_buf[i] = view_buf[i].tolist()            
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
      
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list                        
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list
                k = k + 1        
      
        def Merge(dict1, dict2):        
            res = {**dict1, **dict2} 
            return res
        listofneighbors = Merge(listofneighbors0,listofneighbors1)

        feature_dict = {}
        for i in range(n_group):
            for j in range(len(identity[i])):
                feature_dict[identity[i][j]] = feature_buf[i][j]
        # print(feature_dict)   

        # print(listofneighbors)
        #######
        graph_dict = {}
        for key in listofneighbors:   
            neighbors = listofneighbors.get(key)   
            # print(neighbors)
            new_feature_graph = []
            if len(neighbors) <= 0:  
                # a.append(key)
                graph_dict[key] = neighbors
            else:
                for m in range(len(neighbors)):   
                    neighbor_idx = neighbors[m]
                    # print(new_feature_buf[neighbor_idx])
                    new_feature_graph1 = feature_dict.get(neighbor_idx)  
                    # print(new_feature_graph1)  
                    new_feature_graph.append(new_feature_graph1)  
                new_feature_graph.insert(0,feature_dict.get(key)) 
                num_graph = len(new_feature_graph)
                # print(num_graph,hid_size)
                gt = Gt(num_graph,hid_size)
                prev_hid = torch.zeros(2,num_graph, hid_size) 
                if step_ct == 0:
                    prev_hid = gt.init_hidden(batch_size)  
                new_feature_graph = torch.tensor(new_feature_graph)
                x = [new_feature_graph, prev_hid] 
                adj_id,prev_hid = gt.forward(x)  
                adj=adj_id.tolist()       
                # print(adj[0])    
                # neighbors_result=neighbors
                # neighbors_result.insert(0,key)
                # print(neighbors_result)
                res_neighbors=[]
                for i in range(len(adj[0])-1):
                    if adj[0][i+1] == 1:
                        res_neighbors.append(neighbors[i])
                graph_dict[key] = res_neighbors        
                #############################
                # if len(res_neighbors) == 0:
                #     graph_dict[key] = res_neighbors  
            
                # print(np.array(new_feature_graph).shape)   
                ###########



        graph_dict = dict(sorted(graph_dict.items()))  
        # print(graph_dict)

        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                # new_list = []
                temp_list = graph_dict.get(identity[i][j]) 
                # # print(temp_list)    
                # for l in range(len(temp_list)):
                #     new_list.append(totalagentids[temp_list[l]])
                if i == 0:
                    listofneighbors_graph0[identity[i][j]] = temp_list
                else:
                    listofneighbors_graph1[identity[i][j]] = temp_list
                # print(listofneighbors_graph0)

        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        
        for i in range(n_group):
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)
        
        flag = 1
        
        
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors_graph0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors_graph1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        


        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]        
        if train:
            models[1].flush_buffer(**buffer)
        
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])

        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
                new_act_prob[i].append(temp_var[0])
            
            former_act_prob[i] = np.asarray(new_act_prob[i])
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_num = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])
    
    
    return max_nums, nums, mean_rewards, total_rewards



def battle(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""

    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    nums_per = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    categories = []
    alphas = {}
    for i in range(n_action[0]):
        categories.append(i)
    for i in range(sum(nums)):
        tempe_list = [1] * len(categories)
        alphas[i] = tempe_list
    
    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])
        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                temp_list = env.get_neighbors(k,totalagentpositions)
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list

                
                k = k + 1
        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements))
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements))
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0:
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
                former_act_prob2[i] = np.tile(former_act_prob2[i], (len(state[i][0]), 1))
            if i == pomfq_position:
                acts[i] = models[i].act(state=state[i], prob=former_act_prob2[i], eps=eps, ids = ids[i])
            else:
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])  
       


        flag = 1
        act_dict = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }

        buffer['prob'] = former_act_prob[0]
        
        if train:
            models[0].flush_buffer(**buffer)
        
        buffer = {
            'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
            'alives': alives[1], 'ids': ids[1]
        }

        buffer['prob'] = former_act_prob[1]
        
        if train:
            models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                new_act_prob[i].append(temp_var[0])

            former_act_prob[i] = np.asarray(new_act_prob[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                length = len(temp_list)
                tempe_list = alphas[ids[i][j]]
                for k in range(length):
                    tempe_list[temp_list[k]] = tempe_list[temp_list[k]] + 1
                tempe_list = np.asarray(tempe_list)
                alphas[ids[i][j]] = tempe_list
                new_samples = dirichlet.rvs(tempe_list, size = 100, random_state = 1)
                new_mean = np.mean(new_samples, axis = 0)
                new_act_prob[i].append(new_mean)

            former_act_prob2[i] = np.asarray(new_act_prob[i])
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_nums = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))
    
    if train:
        models[0].train()
        models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
    
    
def battle2(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #This function pertains to the faceoff contests  


    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]
    view_buf = [None for _ in range(n_group)]
    x = [None for _ in range(n_group)]
    adj = [None for _ in range(n_group)]
    feature_buf = [None for _ in range(n_group)]
    prev_hid =[None for _ in range(n_group)]
    hid_size = 64
    batch_size = 1
    detach_gap = 10000
    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    categories = []
    alphas = {}
    for i in range(n_action[0]):
        categories.append(i)
    for i in range(sum(nums)):
        tempe_list = [1] * len(categories)
        alphas[i] = tempe_list

    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        listofneighbors_graph0 = {}
        listofneighbors_graph1 = {}              
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            view_buf[i], feature_buf[i] = env.get_observation_whole(handles[i])
            feature_buf[i] = torch.tensor(feature_buf[i])
            feature_buf[i] = feature_buf[i].tolist()    
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])

        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                new_list1 = []
                group_list1 = []
                temp_list = env.get_neighbors(k,totalagentpositions)   #pomfq
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])                   
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list                
                k = k + 1

        def Merge(dict1, dict2):        
            res = {**dict1, **dict2} 
            return res
        listofneighbors = Merge(listofneighbors0,listofneighbors1)

        feature_dict = {}
        for i in range(n_group):
            for j in range(len(identity[i])):
                feature_dict[identity[i][j]] = feature_buf[i][j]
        # print(feature_dict)  
        #######
        graph_dict = {}
        for key in listofneighbors0:  
            neighbors = listofneighbors0.get(key)  
            # print(neighbors)
            new_feature_graph = []
            if len(neighbors) <= 5:  
                # a.append(key)
                graph_dict[key] = neighbors
            else:
                for m in range(len(neighbors)):   
                    neighbor_idx = neighbors[m]
                    # print(new_feature_buf[neighbor_idx])
                    new_feature_graph1 = feature_dict.get(neighbor_idx)  
                    # print(new_feature_graph1)  
                    new_feature_graph.append(new_feature_graph1)   
                new_feature_graph.insert(0,feature_dict.get(key)) 
                num_graph = len(new_feature_graph)
                # print(num_graph,hid_size)
                gt = Gt(num_graph,hid_size)
                prev_hid = torch.zeros(2,num_graph, hid_size) 
                if step_ct == 0:
                    prev_hid = gt.init_hidden(batch_size)  
                new_feature_graph = torch.tensor(new_feature_graph)
                x = [new_feature_graph, prev_hid] 
                adj_id,prev_hid = gt.forward(x)  
                adj=adj_id.tolist()       
                # print(adj[0])   
                # neighbors_result=neighbors
                # neighbors_result.insert(0,key)
                # print(neighbors_result)
                res_neighbors=[]
                for i in range(len(adj[0])-1):
                    if adj[0][i+1] == 1:
                        res_neighbors.append(neighbors[i])
                graph_dict[key] = res_neighbors        
                #############################
                # if len(res_neighbors) == 0:
                #     graph_dict[key] = res_neighbors  
            
                # print(np.array(new_feature_graph).shape)  
                ###########

           
        graph_dict = dict(sorted(graph_dict.items())) 

        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                # new_list = []
                temp_list = graph_dict.get(identity[i][j]) 
                # # print(temp_list)    
                # for l in range(len(temp_list)):
                #     new_list.append(totalagentids[temp_list[l]])
                if i == 0:
                    listofneighbors_graph0[identity[i][j]] = temp_list
                # else:
                #     listofneighbors_graph1[identity[i][j]] = temp_list
                # print(listofneighbors_graph0)




        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements))  #gamfq:state[0]
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements)) #pomfq:state[1]
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))    #gamfq:former_act_prob
                former_act_prob2[i] = np.tile(former_act_prob2[i], (len(state[i][0]), 1))    #pomfq:former_act_prob2
            if i == 1: #i=1
                acts[i] = models[i].act(state=state[i], prob=former_act_prob2[i], eps=eps, ids = ids[i])  #pomfq:act[1]
            else:
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])  #gamfq:act[0]
        
        flag = 1
        act_dict = {}
        # act_dict1 = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]
                # act_dict1[ids[i][j]] = acts1[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
        act_largedict2 = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors_graph0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict2[ids[i][j]] = new_list
                # else:
                #     new_list = []
                #     temp_list = listofneighbors_graph1[ids[i][j]]
                #     for l in range(len(temp_list)):
                #         s = act_dict[temp_list[l]]
                #         new_list.append(s)
                #     act_largedict2[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # buffer = {
        #     'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
        #     'alives': alives[0], 'ids': ids[0]
        # }

        # buffer['prob'] = former_act_prob[0]
        
        # if train:
        #     models[0].flush_buffer(**buffer)
        
        # buffer = {
        #     'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
        #     'alives': alives[1], 'ids': ids[1]
        # }

        # buffer['prob'] = former_act_prob2[1]
        
        # if train:
        #     models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                if i == 0:
                    temp_list = act_largedict2[ids[i][j]]
                    if(len(temp_list) == 0):
                        temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                    else:
                        temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                    new_act_prob[i].append(temp_var[0])

                    former_act_prob[i] = np.asarray(new_act_prob[i])

        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                length = len(temp_list)
                tempe_list = alphas[ids[i][j]]
                for k in range(length):
                    tempe_list[temp_list[k]] = tempe_list[temp_list[k]] + 1
                tempe_list = np.asarray(tempe_list)
                alphas[ids[i][j]] = tempe_list
                new_samples = dirichlet.rvs(tempe_list, size = 100, random_state = 1)
                new_mean = np.mean(new_samples, axis = 0)
                new_act_prob[i].append(new_mean)

            former_act_prob2[i] = np.asarray(new_act_prob[i])
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_nums = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    # if train:
    #     models[0].train()
    #     models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def battle2_change(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #This function pertains to the faceoff contests  


    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]
    view_buf = [None for _ in range(n_group)]
    x = [None for _ in range(n_group)]
    adj = [None for _ in range(n_group)]
    feature_buf = [None for _ in range(n_group)]
    prev_hid =[None for _ in range(n_group)]
    hid_size = 64
    batch_size = 1
    detach_gap = 10000
    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    categories = []
    alphas = {}
    for i in range(n_action[0]):
        categories.append(i)
    for i in range(sum(nums)):
        tempe_list = [1] * len(categories)
        alphas[i] = tempe_list

    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        listofneighbors_graph0 = {}
        listofneighbors_graph1 = {}              
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            view_buf[i], feature_buf[i] = env.get_observation_whole(handles[i])
            feature_buf[i] = torch.tensor(feature_buf[i])
            feature_buf[i] = feature_buf[i].tolist()    
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])

        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                new_list1 = []
                group_list1 = []
                temp_list = env.get_neighbors(k,totalagentpositions)   #pomfq
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])                   
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list                
                k = k + 1

        def Merge(dict1, dict2):        
            res = {**dict1, **dict2} 
            return res
        listofneighbors = Merge(listofneighbors0,listofneighbors1)

        feature_dict = {}
        for i in range(n_group):
            for j in range(len(identity[i])):
                feature_dict[identity[i][j]] = feature_buf[i][j]
        # print(feature_dict)   
        ######
        graph_dict = {}
        for key in listofneighbors1:   
            neighbors = listofneighbors1.get(key)  
            # print(neighbors)
            new_feature_graph = []
            if len(neighbors) <= 5:  
                # a.append(key)
                graph_dict[key] = neighbors
            else:
                for m in range(len(neighbors)):  
                    neighbor_idx = neighbors[m]
                    # print(new_feature_buf[neighbor_idx])
                    new_feature_graph1 = feature_dict.get(neighbor_idx)  
                    # print(new_feature_graph1)  
                    new_feature_graph.append(new_feature_graph1)   
                new_feature_graph.insert(0,feature_dict.get(key))
                num_graph = len(new_feature_graph)
                # print(num_graph,hid_size)
                gt = Gt(num_graph,hid_size)
                prev_hid = torch.zeros(2,num_graph, hid_size) 
                if step_ct == 0:
                    prev_hid = gt.init_hidden(batch_size)  
                new_feature_graph = torch.tensor(new_feature_graph)
                x = [new_feature_graph, prev_hid] 
                adj_id,prev_hid = gt.forward(x)  
                adj=adj_id.tolist()       
                # print(adj[0])   
                # neighbors_result=neighbors
                # neighbors_result.insert(0,key)
                # print(neighbors_result)
                res_neighbors=[]
                for i in range(len(adj[0])-1):
                    if adj[0][i+1] == 1:
                        res_neighbors.append(neighbors[i])
                graph_dict[key] = res_neighbors        
                #############################
                # if len(res_neighbors) == 0:
                #     graph_dict[key] = res_neighbors  
            
                # print(np.array(new_feature_graph).shape)  
                ###########

             
        graph_dict = dict(sorted(graph_dict.items()))  

        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                # new_list = []
                temp_list = graph_dict.get(identity[i][j]) 
                # # print(temp_list)    
                # for l in range(len(temp_list)):
                #     new_list.append(totalagentids[temp_list[l]])
                if i == 1:
                    listofneighbors_graph1[identity[i][j]] = temp_list
                # else:
                #     listofneighbors_graph1[identity[i][j]] = temp_list
                # print(listofneighbors_graph0)




        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 1:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements))  #gamfq:state[0]
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements)) #pomfq:state[1]
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))    #gamfq:former_act_prob
                former_act_prob2[i] = np.tile(former_act_prob2[i], (len(state[i][0]), 1))    #pomfq:former_act_prob2
            if i == 0: #i=1
                acts[i] = models[i].act(state=state[i], prob=former_act_prob2[i], eps=eps, ids = ids[i])  #pomfq:act[1]
            else:
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])  #gamfq:act[0]
        
        flag = 1
        act_dict = {}
        # act_dict1 = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]
                # act_dict1[ids[i][j]] = acts1[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
        act_largedict2 = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 1:
                    new_list = []
                    temp_list = listofneighbors_graph1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict2[ids[i][j]] = new_list
                # else:
                #     new_list = []
                #     temp_list = listofneighbors_graph1[ids[i][j]]
                #     for l in range(len(temp_list)):
                #         s = act_dict[temp_list[l]]
                #         new_list.append(s)
                #     act_largedict2[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # buffer = {
        #     'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
        #     'alives': alives[0], 'ids': ids[0]
        # }

        # buffer['prob'] = former_act_prob[0]
        
        # if train:
        #     models[0].flush_buffer(**buffer)
        
        # buffer = {
        #     'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
        #     'alives': alives[1], 'ids': ids[1]
        # }

        # buffer['prob'] = former_act_prob2[1]
        
        # if train:
        #     models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                if i == 1:
                    temp_list = act_largedict2[ids[i][j]]
                    if(len(temp_list) == 0):
                        temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                    else:
                        temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                    new_act_prob[i].append(temp_var[0])

                    former_act_prob[i] = np.asarray(new_act_prob[i])

        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                length = len(temp_list)
                tempe_list = alphas[ids[i][j]]
                for k in range(length):
                    tempe_list[temp_list[k]] = tempe_list[temp_list[k]] + 1
                tempe_list = np.asarray(tempe_list)
                alphas[ids[i][j]] = tempe_list
                new_samples = dirichlet.rvs(tempe_list, size = 100, random_state = 1)
                new_mean = np.mean(new_samples, axis = 0)
                new_act_prob[i].append(new_mean)

            former_act_prob2[i] = np.asarray(new_act_prob[i])
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_nums = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    # if train:
    #     models[0].train()
    #     models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def battle3(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #This function pertains to the faceoff contests  


    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]
    view_buf = [None for _ in range(n_group)]
    x = [None for _ in range(n_group)]
    adj = [None for _ in range(n_group)]
    feature_buf = [None for _ in range(n_group)]
    prev_hid =[None for _ in range(n_group)]
    hid_size = 64
    batch_size = 1
    detach_gap = 10000
    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    # categories = []
    # alphas = {}
    # for i in range(n_action[0]):
    #     categories.append(i)
    # for i in range(sum(nums)):
    #     tempe_list = [1] * len(categories)
    #     alphas[i] = tempe_list

    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        listofneighbors_graph0 = {}
        listofneighbors_graph1 = {}              
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            view_buf[i], feature_buf[i] = env.get_observation_whole(handles[i])
            feature_buf[i] = torch.tensor(feature_buf[i])
            feature_buf[i] = feature_buf[i].tolist()    
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])

        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                new_list1 = []
                group_list1 = []
                temp_list = env.get_neighbors(k,totalagentpositions)   #pomfq
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])                   
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list                
                k = k + 1

        def Merge(dict1, dict2):        
            res = {**dict1, **dict2} 
            return res
        listofneighbors = Merge(listofneighbors0,listofneighbors1)

        feature_dict = {}
        for i in range(n_group):
            for j in range(len(identity[i])):
                feature_dict[identity[i][j]] = feature_buf[i][j]
        # print(feature_dict)  
        #######
        graph_dict = {}
        for key in listofneighbors0:  
            neighbors = listofneighbors0.get(key)  
            # print(neighbors)
            new_feature_graph = []
            if len(neighbors) <=5: 
                # a.append(key)
                graph_dict[key] = neighbors
            else:
                for m in range(len(neighbors)):   
                    neighbor_idx = neighbors[m]
                    # print(new_feature_buf[neighbor_idx])
                    new_feature_graph1 = feature_dict.get(neighbor_idx)  
                    # print(new_feature_graph1)  
                    new_feature_graph.append(new_feature_graph1)   
                new_feature_graph.insert(0,feature_dict.get(key)) 
                num_graph = len(new_feature_graph)
                # print(num_graph,hid_size)
                gt = Gt(num_graph,hid_size)
                prev_hid = torch.zeros(2,num_graph, hid_size) 
                if step_ct == 0:
                    prev_hid = gt.init_hidden(batch_size)  
                new_feature_graph = torch.tensor(new_feature_graph)
                x = [new_feature_graph, prev_hid] 
                adj_id,prev_hid = gt.forward(x)  
                adj=adj_id.tolist()       
                # print(adj[0])   
                # neighbors_result=neighbors
                # neighbors_result.insert(0,key)
                # print(neighbors_result)
                res_neighbors=[]
                for i in range(len(adj[0])-1):
                    if adj[0][i+1] == 1:
                        res_neighbors.append(neighbors[i])
                graph_dict[key] = res_neighbors        
                #############################
                # if len(res_neighbors) == 0:
                #     graph_dict[key] = res_neighbors 
            
                # print(np.array(new_feature_graph).shape)   
                ###########

                
        graph_dict = dict(sorted(graph_dict.items()))  

        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                # new_list = []
                temp_list = graph_dict.get(identity[i][j]) 
                # # print(temp_list)    
                # for l in range(len(temp_list)):
                #     new_list.append(totalagentids[temp_list[l]])
                if i == 0:
                    listofneighbors_graph0[identity[i][j]] = temp_list
                # else:
                #     listofneighbors_graph1[identity[i][j]] = temp_list
                # print(listofneighbors_graph0)




        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 0:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements))  #gamfq:state[0]
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements)) #pomfq:state[1]
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))    #gamfq:former_act_prob
                former_act_prob2[i] = np.tile(former_act_prob2[i], (len(state[i][0]), 1))    #mfq:former_act_prob2
            if i == 1: #i=1
                acts[i] = models[i].act(state=state[i], prob=former_act_prob2[i], eps=eps, ids = ids[i])  #mfq:act[1]
            else:
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])  #gamfq:act[0]
        
        flag = 1
        act_dict = {}
        # act_dict1 = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]
                # act_dict1[ids[i][j]] = acts1[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
        act_largedict2 = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors_graph0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict2[ids[i][j]] = new_list
                # else:
                #     new_list = []
                #     temp_list = listofneighbors_graph1[ids[i][j]]
                #     for l in range(len(temp_list)):
                #         s = act_dict[temp_list[l]]
                #         new_list.append(s)
                #     act_largedict2[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # buffer = {
        #     'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
        #     'alives': alives[0], 'ids': ids[0]
        # }

        # buffer['prob'] = former_act_prob[0]
        
        # if train:
        #     models[0].flush_buffer(**buffer)
        
        # buffer = {
        #     'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
        #     'alives': alives[1], 'ids': ids[1]
        # }

        # buffer['prob'] = former_act_prob2[1]
        
        # if train:
        #     models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                if i == 0:
                    temp_list = act_largedict2[ids[i][j]]
                    if(len(temp_list) == 0):
                        temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                    else:
                        temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                    new_act_prob[i].append(temp_var[0])

                    former_act_prob[i] = np.asarray(new_act_prob[i])

        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                new_act_prob[i].append(temp_var[0])

            former_act_prob2[i] = np.asarray(new_act_prob[i])
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_nums = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    # if train:
    #     models[0].train()
    #     models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def battle3_change(env, n_round, map_size, max_steps, handles, models, pomfq_position, print_every, eps=1.0, render=False, train=False):
    """play a round and train"""
    #This function pertains to the faceoff contests  


    env.reset()
    food_handle = handles[0]
    handles = handles[1:]
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]
    hps = [None for _ in range(n_group)]
    view_buf = [None for _ in range(n_group)]
    x = [None for _ in range(n_group)]
    adj = [None for _ in range(n_group)]
    feature_buf = [None for _ in range(n_group)]
    prev_hid =[None for _ in range(n_group)]
    hid_size = 64
    batch_size = 1
    detach_gap = 10000
    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]
    
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]
    identity = [[] for _ in range(n_group)]
    positions = [[] for _ in range(n_group)]
    maxelements = 20
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    former_act_prob2 = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))] 
    flag = 0
    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
    # categories = []
    # alphas = {}
    # for i in range(n_action[0]):
    #     categories.append(i)
    # for i in range(sum(nums)):
    #     tempe_list = [1] * len(categories)
    #     alphas[i] = tempe_list

    while not done and step_ct < max_steps:
        listofneighbors0 = {}
        listofneighbors1 = {}
        grouplist0 = {}
        grouplist1 = {}
        listofneighbors_graph0 = {}
        listofneighbors_graph1 = {}              
        totalagentpositions = []
        totalagentids = []
        group_dict = {}
        for i in range(n_group):
            positions[i] = env.get_pos(handles[i])
            identity[i] = env.get_agent_id(handles[i])
            view_buf[i], feature_buf[i] = env.get_observation_whole(handles[i])
            feature_buf[i] = torch.tensor(feature_buf[i])
            feature_buf[i] = feature_buf[i].tolist()    
            for j in range(len(positions[i])):
                group_dict[identity[i][j]] = i
                totalagentpositions.append(positions[i][j])
                totalagentids.append(identity[i][j])

        k = 0
        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                new_list = []
                group_list = []
                new_list1 = []
                group_list1 = []
                temp_list = env.get_neighbors(k,totalagentpositions)   #pomfq
                for l in range(len(temp_list)):
                    new_list.append(totalagentids[temp_list[l]])                   
                    group_list.append(group_dict[totalagentids[temp_list[l]]])
                    
                if i == 0:
                    listofneighbors0[identity[i][j]] = new_list
                    grouplist0[identity[i][j]] = group_list
                else:
                    listofneighbors1[identity[i][j]] = new_list
                    grouplist1[identity[i][j]] = group_list                
                k = k + 1

        def Merge(dict1, dict2):        
            res = {**dict1, **dict2} 
            return res
        listofneighbors = Merge(listofneighbors0,listofneighbors1)

        feature_dict = {}
        for i in range(n_group):
            for j in range(len(identity[i])):
                feature_dict[identity[i][j]] = feature_buf[i][j]
        # print(feature_dict)   
        #######
        graph_dict = {}
        for key in listofneighbors1:   
            neighbors = listofneighbors1.get(key)  
            # print(neighbors)
            new_feature_graph = []
            if len(neighbors) <= 5: 
                # a.append(key)
                graph_dict[key] = neighbors
            else:
                for m in range(len(neighbors)):   
                    neighbor_idx = neighbors[m]
                    # print(new_feature_buf[neighbor_idx])
                    new_feature_graph1 = feature_dict.get(neighbor_idx) 
                    # print(new_feature_graph1)  
                    new_feature_graph.append(new_feature_graph1)   
                new_feature_graph.insert(0,feature_dict.get(key))
                num_graph = len(new_feature_graph)
                # print(num_graph,hid_size)
                gt = Gt(num_graph,hid_size)
                prev_hid = torch.zeros(2,num_graph, hid_size) 
                if step_ct == 0:
                    prev_hid = gt.init_hidden(batch_size)  
                new_feature_graph = torch.tensor(new_feature_graph)
                x = [new_feature_graph, prev_hid] 
                adj_id,prev_hid = gt.forward(x)  
                adj=adj_id.tolist()       
                # print(adj[0])  
                # neighbors_result=neighbors
                # neighbors_result.insert(0,key)
                # print(neighbors_result)
                res_neighbors=[]
                for i in range(len(adj[0])-1):
                    if adj[0][i+1] == 1:
                        res_neighbors.append(neighbors[i])
                graph_dict[key] = res_neighbors        
                #############################
                # if len(res_neighbors) == 0:
                #     graph_dict[key] = res_neighbors  
            
                # print(np.array(new_feature_graph).shape)  
                ###########
           
        graph_dict = dict(sorted(graph_dict.items())) 

        for i in range(n_group):
            for j in range(env.get_num(handles[i])):
                # new_list = []
                temp_list = graph_dict.get(identity[i][j]) 
                # # print(temp_list)    
                # for l in range(len(temp_list)):
                #     new_list.append(totalagentids[temp_list[l]])
                if i == 1:
                    listofneighbors_graph1[identity[i][j]] = temp_list
                # else:
                #     listofneighbors_graph1[identity[i][j]] = temp_list
                # print(listofneighbors_graph0)




        # take actions for every model
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if i == 1:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors1, grouplist1, maxelements))  #gamfq:state[0]
            else:
                state[i] = list(env.get_observation_gather(food_handle, handles[i], handles, listofneighbors0, grouplist0, maxelements)) #pomfq:state[1]
        
        
        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
            if flag == 0: 
                former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))    #gamfq:former_act_prob
                former_act_prob2[i] = np.tile(former_act_prob2[i], (len(state[i][0]), 1))    #mfq:former_act_prob2
            if i == 0: #i=1
                acts[i] = models[i].act(state=state[i], prob=former_act_prob2[i], eps=eps, ids = ids[i])  #mfq:act[1]
            else:
                acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps, ids = ids[i])  #gamfq:act[0]
        
        flag = 1
        act_dict = {}
        # act_dict1 = {}
        for i in range(n_group):
            for j in range(len(ids[i])):
                act_dict[ids[i][j]] = acts[i][j]
                # act_dict1[ids[i][j]] = acts1[i][j]

        act_largedict = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 0:
                    new_list = []
                    temp_list = listofneighbors0[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
                else:
                    new_list = []
                    temp_list = listofneighbors1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict[ids[i][j]] = new_list
        act_largedict2 = {}
        for i in range(n_group): 
            for j in range(len(ids[i])):
                if i == 1:
                    new_list = []
                    temp_list = listofneighbors_graph1[ids[i][j]]
                    for l in range(len(temp_list)):
                        s = act_dict[temp_list[l]]
                        new_list.append(s)
                    act_largedict2[ids[i][j]] = new_list
                # else:
                #     new_list = []
                #     temp_list = listofneighbors_graph1[ids[i][j]]
                #     for l in range(len(temp_list)):
                #         s = act_dict[temp_list[l]]
                #         new_list.append(s)
                #     act_largedict2[ids[i][j]] = new_list

        for i in range(n_group):
            env.set_action(handles[i], acts[i])
        # simulate one step
        done = env.step()
        

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # buffer = {
        #     'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
        #     'alives': alives[0], 'ids': ids[0]
        # }

        # buffer['prob'] = former_act_prob[0]
        
        # if train:
        #     models[0].flush_buffer(**buffer)
        
        # buffer = {
        #     'state': state[1], 'acts': acts[1], 'rewards': rewards[1],
        #     'alives': alives[1], 'ids': ids[1]
        # }

        # buffer['prob'] = former_act_prob2[1]
        
        # if train:
        #     models[1].flush_buffer(**buffer)
        # clear dead agents
        env.clear_dead()

        for i in range(n_group):
            ids[i] = env.get_agent_id(handles[i])
        
        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                if i == 1:
                    temp_list = act_largedict2[ids[i][j]]
                    if(len(temp_list) == 0):
                        temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                    else:
                        temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                    new_act_prob[i].append(temp_var[0])

                    former_act_prob[i] = np.asarray(new_act_prob[i])

        new_act_prob = [[] for i in range(n_group)]
        for i in range(n_group):
            for j in range(len(ids[i])):
                temp_list = act_largedict[ids[i][j]]
                if(len(temp_list) == 0):
                    temp_var = np.zeros((1, env.get_action_space(handles[0])[0]))
                else:
                    temp_var = np.mean(list(map(lambda x: np.eye(n_action[i])[x], temp_list)), axis=0, keepdims=True)
            
                new_act_prob[i].append(temp_var[0])

            former_act_prob2[i] = np.asarray(new_act_prob[i])
        
        # stat info
        nums = [env.get_num(handle) for handle in handles]
        food_nums = env.get_num(food_handle)
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()


        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    # if train:
    #     models[0].train()
    #     models[1].train()

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards

 
    
    
