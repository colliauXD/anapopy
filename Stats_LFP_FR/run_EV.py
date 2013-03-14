import numpy as np
import pylab as pl
import utils_SLF as us
#from NeuroTools.signals import spikes as spk

def plot_raster_EV(s,sig,R_time,thr,lm,figfile=""):   
   f=pl.figure()
   gs=pl.matplotlib.gridspec.GridSpec(2,1,height_ratios=[3,1])
   ax1=f.add_subplot(gs[0])
   pl.bar( R_time[lm],np.max(s.id_list)*np.ones(len(lm)),color="k",linewidth=0,width=dt,alpha=0.2,align="center")
   s.raster_plot(display=ax1,kwargs={"color":"k"})
   ax1.set_ylabel("Neuron ID")
   ax2=f.add_subplot(gs[1])
   ax2.plot(R_time,sig,"k")
   ax2.plot(R_time[lm],sig[lm],"ro")
   ax2.fill_between([s.t_start-100,s.t_stop+100],[0,0],[thr,thr],alpha=0.2)
   ax2.set_xlim([s.t_start-100,s.t_stop+100])
   ax2.set_ylabel("Pop. firing rate [Hz]")
   pl.savefig(figfile+"_ras_EV.pdf")

dt=10
fname="col_ab.spk"

s, sig, lm, R_time=us.get_EV(dt,fname)
thr=np.mean(sig)
lm=us.filter_thresh(lm,sig,thr)

plot_raster_EV(s,sig,R_time,thr,lm,"col_ab")
