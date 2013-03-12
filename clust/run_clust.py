import numpy as np
import pylab as pl
import utils_clust as uc
from sklearn.cluster import AffinityPropagation
from NeuroTools.signals import spikes as spk

def aff_prop_scan(cc,pref_range=None,N=150):
   """
   Scan various (homogeneous) preference values
   return preferences, #clusters  
   """
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

def plot_scan(p,l,fig_name="scan_p.eps"):
    """
    plot the number of clusters 
    """
    pl.plot(p,l,"ko")
    pl.title(r"$N_{cluster}$")
    pl.xlabel("preference")
    pl.savefig(fig_name)
    return 0

def plot_clust_ord(s,labs,fname="clust_ord.eps"):
   plot_idx=0
   size_Cs=np.zeros(len(cents))
   cols=np.random.random([len(cents),3])
   k=0
   for i in range(len(cents)):
      cl_idx=s.id_list[np.where(labs==i)[0]]
      size_Cs[i]=len(cl_idx)
      for j in cl_idx:
         st=s.spiketrains[j].spike_times
         pl.plot(st,plot_idx*np.ones(len(st)),",",color=cols[i])
         plot_idx+=1
   pl.xlabel("Time (ms)")
   pl.ylabel("Neurons ID")
   a = pl.axes([.65, .6, .2, .2], axisbg="0.5")
   pl.bar(range(len(size_Cs)),size_Cs,color=cols)
   pl.title('Cluster sizes')
   pl.setp(a, xticks=[], yticks=[0,np.max(size_Cs)])
   pl.savefig(fname)
   return 0
   
NS=500
fname="test.spk"
#fname="col_ab.spk"
s=uc.get_N_non_silent(fname,NS)
NS=len(s.id_list)
cc=uc.get_cc_mat(s,NS,tb=15)
cc+=np.random.normal(0,0.01*np.abs(np.min(cc-1)),len(cc))
p,l=aff_prop_scan(cc-1,[3*np.min(cc-1),0.01],N=150)
plot_scan(p,l,"scan_trans.eps")
pl.clf()
pref=np.median(cc-1)
af=AffinityPropagation(max_iter=100,copy=True).fit(cc-1,pref)
cents=af.cluster_centers_indices_
labs=af.labels_
plot_clust_ord(s,labs,fname="clust_ord.eps")
