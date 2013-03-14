import numpy as np
from NeuroTools.signals import spikes as spk

def filter_thresh(loc_ev,sig,thr):
   return loc_ev[sig[loc_ev]>thr]
   
def get_EV(dt,fname):
   s=spk.load_spikelist("col_ab.spk")
   sig=s.firing_rate(dt,average=True)
   loc_ev = (np.diff(np.sign(np.diff(sig))) < 0).nonzero()[0]+1 
   R_time=np.linspace(s.t_start,s.t_stop,len(sig))
   return s, sig, loc_ev, R_time
