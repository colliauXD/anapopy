from NeuroTools import tisean as tsn
import numpy as np
import pylab as pl

def get_delay(n,input_file,fname="test",plotit=True):
   cmd="mutual"
   outputfile="lorenz.MI"
   loc_mins=[]
   col=["","chartreuse","plum","coral"] 
   for i in range(1,n+1):
      options="-c%s -b64 -D60"%i
      tsn.foperation(cmd, options, inputfile, outputfile)
      MI=np.loadtxt(outputfile)
      if plotit: pl.plot(MI[:,0],MI[:,1],col[i],linewidth=2.5,label="%s th var"%i)
      MIp=np.diff(MI[:,1])
      a = np.diff(np.sign(np.diff(MI[:,1]))).nonzero()[0] + 1 
      loc_min = (np.diff(np.sign(np.diff(MI[:,1]))) > 0).nonzero()[0] + 1 
      loc_mins.append(loc_min)
      if plotit: pl.plot(MI[loc_min,0],MI[loc_min,1], "o", color=col[i])
   if plotit:
      pl.legend()
      pl.xlabel(r"delay $\tau$")
      pl.ylabel(r"$I(x_t,x_{t+\tau})$")
      pl.savefig(fname+".png")
      pl.clf()
   return max(loc_mins)[0]

def get_emb_dim(delay,factor,m,M,inputfile,fname="test",plotit=True):
   cmd="false_nearest"
   outputfile=fname+".fnn"
   options="-m%s -M1,%s -d%s -f%s -t2000"%(m,M,delay,factor)
   tsn.foperation(cmd, options, inputfile, outputfile)
   a=np.loadtxt(outputfile)
   if plotit: pl.plot(a[:,0],a[:,1],"o",color="k")
   if plotit:
      pl.savefig(fname+"png")
      pl.clf()
   ds=a[np.where(a[:,1]<1e-4)[0],0]
   return ds[0]  

def get_maxlyap(delay,dt,ws,m,M,inputfile,plotit="Lyap_k"):
   options="-x300000 -m%s -M%s -d%s -s%s -r.1"%(m,M,delay,ws)
   outputfile="Lorenz.lyap"
   cmd="lyap_k"
   tsn.foperation(cmd, options, inputfile, outputfile)

   #Fit linear part
   l=np.loadtxt("Lorenz.lyap")     
   s=l.shape[0]
   ls0=np.transpose(l[:,0].reshape(s/(ws+1),ws+1))
   ls1=np.transpose(l[:,1].reshape(s/(ws+1),ws+1))
   
   import scipy.optimize as opt
   LEs=[]
   fitfunc = lambda p, x: p[0]*x + p[1] # Target function
   errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
   for i in range(s/(ws+1)):
      p0 = [10, -5] # Initial guess for the parameters
      p1, success = opt.leastsq(errfunc, p0[:], args=(dt*ls0[100:400,i], ls1[100:400,i]))
      print success,p1[0] 
      LEs.append(p1[0])
      if plotit:
         pl.plot(dt*ls0[100:400,i],fitfunc(p1,dt*ls0[100:400,i]),"r")
         pl.plot(dt*ls0[100:400,i],ls1[100:400,i],"k,")       
   if plotit:
      pl.title("Max Lyapunov exponent: %s"%np.mean(np.array(LEs)))
      pl.savefig("LorLyapExp.png")      
   return np.array(LEs)

n=3
dt=0.01
inputfile="Lorenz.traj"
fname="Lorenz"
m=1
M=5
delay=get_delay(n,inputfile,fname)
factor=20 #Defined heuristically
d=get_emb_dim(delay,factor,m,M,inputfile,fname)
print "Embeding parameter extracted:"
print "delay:", delay, "embedding dimension:", d

"""
ws=1000
get_maxlyap(delay,dt,ws,m,M,inputfile)
N_comp=3
N_emb=1
tw=1000
options="-M3,1 -d1 -N50000"
outputfile="Lorenz.corsum"
cmd="d2"
tsn.foperation(cmd, options, inputfile, outputfile)
cs=np.loadtxt(outputfile+".d2")
cs0=np.transpose(cs[:,0].reshape(len(cs[:,0])/99,99))
cs1=np.transpose(cs[:,1].reshape(len(cs[:,1])/99,99))
"""
