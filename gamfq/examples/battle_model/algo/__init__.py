from . import ac
from . import q_learning
from . import rnnq_learning
AC = ac.ActorCritic
MFAC = ac.MFAC
IL = q_learning.DQN
MFQ = q_learning.MFQ
POMFQ = q_learning.POMFQ
GAMFQ = q_learning.GAMFQ
GAMFQ1 = q_learning.GAMFQ1
rnnIL = rnnq_learning.DQN
rnnMFQ = rnnq_learning.MFQ

def spawn_ai(algo_name, sess, env, handle, human_name, max_steps):
    if algo_name == 'mfq':
        model = MFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'mfac':
        model = MFAC(sess, human_name, handle, env)
    elif algo_name == 'ac':
        model = AC(sess, human_name, handle, env)
    elif algo_name == 'il':
        model = IL(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'rnnIL':
        model = rnnIL(sess, human_name, handle, env, max_steps, memory_size=80000) 
    elif algo_name == 'rnnMFQ':
        model = rnnMFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'pomfq':
        model = POMFQ(sess, human_name, handle, env, max_steps, memory_size=80000)
    elif algo_name == 'gamfq':
        model = GAMFQ(sess, human_name, handle, env, max_steps, memory_size=80000) 
    elif algo_name == 'gamfq1':
        model = GAMFQ1(sess, human_name, handle, env, max_steps, memory_size=80000)                
    return model
