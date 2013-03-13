import numpy as np
import scipy as sc
import pylab as pl
from NeuroTools.signals import spikes as spk

def get_peak_vt(sig,thresh):
   a = np.diff(np.sign(np.diff(MI[:,1]))).nonzero()[0] + 1 
   loc_min = (np.diff(np.sign(np.diff(MI[:,1]))) > 0).nonzero()[0] + 1 
   
s=spk.load_spikelist("col_ab.spk")

sig=s.firing_rate(2,average=True)
thr=2*np.std(sig)
a = np.diff(np.sign(np.diff(sig))).nonzero()[0] + 1 
loc_min = (np.diff(np.sign(np.diff(sig))) > 0).nonzero()[0]  
if thr>0:
   loc_max=loc_min[sig[loc_min]>thr]
pl.plot(sig)
pl.plot(loc_max,3*np.ones(len(loc_max)),".")
#pl.hist(sig,100)
pl.show()
