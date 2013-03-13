from NeuroTools import tisean as tsn
import numpy as np
import pylab as pl
import time

def get_delay(n,inputfile,fname="test",plotit=True):
   cmd="mutual"
   outputfile=fname+".MI"
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
      pl.savefig(fname+"_MI.png")
      pl.clf()
   return max(loc_mins)[0]

def get_emb_dim(delay,factor,m,M,inputfile,fname="test",plotit=True):
   cmd="false_nearest"
   outputfile=fname+".fnn"
   options="-m%s -M1,%s -d%s -f%s -t2000"%(m,M,delay,factor)
   tsn.foperation(cmd, options, inputfile, outputfile)
   a=np.loadtxt(outputfile)
   if plotit: 
      pl.plot(a[:,0],a[:,1],"o",color="k")
      pl.title("Fraction of false n.n.")
      pl.xlabel("Dimension")
      pl.savefig(fname+"_ED.png")
      pl.clf()
   ds=a[np.where(a[:,1]<1e-4)[0],0]
   return ds[0]  

def get_maxlyap(delay,dt,ws,m,M,inputfile,fname="test",plotit=True):
   options="-x300000 -m%s -M%s -d%s -s%s -r.1"%(m,M,delay,ws)
   outputfile=fname+".lyap"
   cmd="lyap_k"
   tsn.foperation(cmd, options, inputfile, outputfile)

   #Fit linear part
   l=np.loadtxt(fname+".lyap")     
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
      pl.savefig(fname+"_MLE.png")      
   return np.array(LEs)

def get_corrsum(delay,d,tw,N,inputfile,fname="test",plotit=True):
   options="-M%s,%s -d%s -t%s -N%s"%(1,int(d),delay,tw,N)
   outputfile=fname+".corsum"
   cmd="d2"
   tsn.foperation(cmd, options, inputfile, outputfile)
   cs=np.loadtxt(outputfile+".c2")
   cs0=np.transpose(cs[:,0].reshape(len(cs[:,0])/100,100))
   cs1=np.transpose(cs[:,1].reshape(len(cs[:,1])/100,100))

   import scipy.optimize as opt
   C2s=[]
   fitfunc = lambda p, x: p[0]*x + p[1] # Target function
   errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function
   for i in range(cs0.shape[1]):
      p0 = [10, -5] # Initial guess for the parameters
      p1, success = opt.leastsq(errfunc, p0[:], args=(np.log(cs0[40:,i]), np.log(cs1[40:,i])))
      print success,p1[0] 
      C2s.append(p1[0])
      if plotit:
         pl.plot(np.log(cs0[40:,i]),fitfunc(p1,np.log(cs0[40:,i])),"r")
         pl.plot(cs0[:,i],cs1[:,i],"k.")       
   if plotit:
      pl.title("Correlation sum: %s"%np.mean(np.array(C2s)))
      pl.savefig(fname+"_C2.png")      
   return np.array(C2s)

def test_Lorenz():
   t0=time.time()
   n=3
   dt=0.01
   inputfile="Lorenz/Lorenz.traj"
   fname="Lorenz/Lorenz"
   m=1
   M=5
   factor=10 #rejection of n.n. factor choosen heuristically (increase if the %fnn doesn t fall to 0)
   ws=1000
   tw=500
   N=50000

   delay=get_delay(n,inputfile,fname)
   d=get_emb_dim(delay,factor,m,M,inputfile,fname)
   LEs=get_maxlyap(delay,dt,ws,m,M,inputfile,fname)
   C2s=get_corrsum(delay,d,tw,N,inputfile,fname)
 
   print "Embeding parameter extracted:"
   print "delay:", delay, "embedding dimension:", d
   print "Lyap. Exp.",LEs
   print "Correlation sum:", C2s[-1]

   print "it took", time.time()-t0

test_Lorenz()
