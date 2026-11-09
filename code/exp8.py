import numpy as np
from lib import *
H=np.load('../results/H.npy')
for a in [0.5,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,np.inf]:
    t=make_table(H,a); print(a, bool(np.all(np.diff(t)>0)), len(t), t.min(), t.max())
