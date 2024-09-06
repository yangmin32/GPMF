                
from scipy.stats import dirichlet
import numpy as np        
                
new_samples = dirichlet.rvs([2,4,7,10,15], size = 100, random_state = 1)
new_mean = np.mean(new_samples, axis = 0)
print(new_mean)