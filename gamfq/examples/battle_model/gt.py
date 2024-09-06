import torch.nn.functional as F
from torch import nn
import torch
from examples.battle_model.gnn_layers import GraphAttention
from functools import reduce
import numpy as np

class Gt(nn.Module):
    '''
    Get an adjacency matrix, which is a communication graph [n * n], an undirected graph
    '''
    def __init__(self,nums,hid_size):
        super(Gt, self).__init__()

        #Parameter settings
        self.nagents = nums   #Number of agents, handle[0] and handle[1]
        self.agent_mask= torch.ones(self.nagents)
        
        self.gat_encoder_out_size = 64   #hidden size of output of the gat encoder
        self.ge_num_heads = 4  #number of heads in the gat encoder
        self.batch_size = 1   #number of steps before each update 
        self.gat_encoder_normalize = False   #Whether to normalize coefficients in the gat encoder (if the input graph is complete, they are already normalized)
        self.hid_size =hid_size
        self.obs_size =34  #Observational dimensions of the environment 1183  34 pursuit 22
        dropout = 0
        negative_slope = 0.2   #0.01,0.1,0.2
        
        #initialize the gat encoder for the Scheduler        
        self.gat_encoder = GraphAttention(self.hid_size, self.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=self.ge_num_heads, self_loop_type=1, average=True, normalize=self.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(self.obs_size, self.hid_size)   #Set the fully connected layer in the network, input observation size, output hidden size

        self.init_hidden(self.batch_size)  

        self.lstm_cell= nn.LSTMCell(self.hid_size, self.hid_size) #LSTM initialization

        #initialize mlp layers for the sub-schedulers 
        self.sub_scheduler_mlp = nn.Sequential(
            nn.Linear(self.gat_encoder_out_size*2, self.gat_encoder_out_size//2),
            nn.ReLU(),
            nn.Linear(self.gat_encoder_out_size//2, self.gat_encoder_out_size//2),
            nn.ReLU(),
            nn.Linear(self.gat_encoder_out_size//2, 2))
        self.sub_scheduler_mlp.apply(self.init_linear)
        self.message_encoder = nn.Linear(self.hid_size, self.hid_size)

        # initialize weights as 0
        self.message_encoder.weight.data.zero_()  


    def forward(self,x):  

        
        # print(obs)
        # obs = torch.tensor(obs)
        obs, extras = x
        hidden_state, cell_state = extras
        encoded_obs = self.obs_encoder(obs)    #coding
        # print(encoded_obs)
        # print(hidden_state)
        # print(cell_state)
        # print(np.array(prev_hid).shape)
        # hidden_state,cell_state = prev_hid
        # hidden_state  = prev_hid
        # cell_state = prev_hid
        # print("encoded_obs shape is {0},hidden_state is {1},cell_state is {2}".format(encoded_obs.shape,hidden_state.shape,cell_state.shape))

        
        # hidden_state, cell_state = self.lstm_cell(encoded_obs.squeeze(), (hidden_state, cell_state))   #LSTM 
        hidden_state, cell_state = self.lstm_cell(encoded_obs, (hidden_state, cell_state))   #LSTM 

        
        # comm: [n * hid_size]
        comm = hidden_state
        comm = self.message_encoder(comm)     
        adj_complete = self.get_complete_graph(self.agent_mask)
        encoded_state = self.gat_encoder(comm, adj_complete) 
        adj = self.sub_scheduler(self.sub_scheduler_mlp, encoded_state, self.agent_mask)   #Computing the adjacency matrix using the scheduler
        
        
        
        return adj,(hidden_state.clone(), cell_state.clone())
       
    def init_hidden(self, batch_size):
        """
        Function to initialize the hidden states and cell states
        """
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True))) 

    def init_linear(self,m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)

    def get_complete_graph(self, agent_mask):
        """
        Function to generate a complete graph, and mask it with agent_mask
        """
        n = self.nagents
        adj = torch.ones(n, n)
        agent_mask = agent_mask.expand(n, n)
        agent_mask_transpose = agent_mask.transpose(0, 1)
        adj = adj * agent_mask * agent_mask_transpose        
        return adj 

    def sub_scheduler(self,sub_scheduler_mlp, hidden_state, agent_mask):
        """
        Function to perform a sub-scheduler

        Arguments: 
            sub_scheduler_mlp (nn.Sequential): the MLP layers in a sub-scheduler 
            hidden_state (tensor): the encoded messages input to the sub-scheduler [n * hid_size]
            agent_mask (tensor): [n * 1]  
            
        Return:
            adj (tensor): a adjacency matrix which is the communication graph [n * n]   
        """

        # hidden_state: [n * hid_size]
        n = self.nagents  #Number of Agents
        hid_size = hidden_state.size(-1)
        # hard_attn_input: [n * n * (2*hid_size)]
        hard_attn_input = torch.cat([hidden_state.repeat(1, n).view(n * n, -1), hidden_state.repeat(n, 1)], dim=1).view(n, -1, 2 * hid_size)
    
        # hard_attn_output: [n * n * 2]
        hard_attn_output = F.gumbel_softmax(0.5*sub_scheduler_mlp(hard_attn_input)+0.5*sub_scheduler_mlp(hard_attn_input.permute(1,0,2)), hard=True)
    
        # hard_attn_output: [n * n * 1]
        hard_attn_output = torch.narrow(hard_attn_output, 2, 1, 1)   #Slice [n * n * 2] along the third dimension 1:2
    
        # agent_mask and agent_mask_transpose: [n * n] 
        agent_mask = agent_mask.expand(n, n)   #Expand n*1 to n*n
        agent_mask_transpose = agent_mask.transpose(0, 1) #agent_mask transpose
    
        # adj: [n * n]
        adj = hard_attn_output.squeeze() * agent_mask * agent_mask_transpose #Compress the dimensions of hard_attn_output, remove the dimension of 1, and compress [n * n * 1] to n*n
        
        return adj   