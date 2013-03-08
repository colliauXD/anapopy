import numpy as np
import pylab as pl
import utils_clust as uc
from sklearn.cluster import AffinityPropagation
#from NeuroTools.signals import spikes as spk

def aff_prop_scan(cc,pref_range=None,N=150):
   if not pref_range: pref_range=[np.min(cc),0.1]
   ps=np.linspace(pref_range[0],pref_range[1],N)
   ls=np.zeros(N)
   af=AffinityPropagation(max_iter=100,copy=True)
   print "scanning preferences for affinity propagation"
   for i in range(N):
      f=af.fit(cc,ps[i])
      ls[i]=len(f.cluster_centers_indices_)
      print ps[i],"done:",ls[i],"centers"
   return ps,ls


def aff_prop_clust(cc,p):
   af=AffinityPropagation(max_iter=100,copy=True)
   return af.fit(cc,p)

NS=500
fname="test.spk"
s=uc.get_N_non_silent(fname,NS)
NS=len(s.id_list)
cc=uc.get_cc_mat(s,NS,tb=5)
cc+=np.random.normal(0,0.01,len(cc))
p,l=aff_prop_scan(cc-1,[1.52*np.min(cc-1),0.1],N=150)
af=aff_prop_clust(cc,np.min(cc))
