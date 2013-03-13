from __future__ import division
import numpy as np

#Lorenz equation

def SJdyn(sJ):
   sJnew=sJ.copy() 
   sJnew[0:3]=[10. * (sJ[1] - sJ[0]), 28. * sJ[0] - sJ[1] - sJ[0]*sJ[2], sJ[0] * sJ[1] - 8./3.0 * sJ[2]]
   for i in xrange(3,6):  sJnew[i]  = -10. * sJ[i] + 10. * sJ[i+3]
   for i in xrange(6,9):  sJnew[i]  = 28.*sJ[i-3]-sJ[2]*sJ[i-3]-sJ[i]-sJ[0]*sJ[i+3]
   for i in xrange(9,12): sJnew[i] = sJ[1]*sJ[i-6] + sJ[0]*sJ[i-3] - 8./3.0 * sJ[i]
   return sJnew

def iterate(sJ, t_it, ts, t_store):
  sList = [] #State history
  t = 0
  sJnew=np.zeros(12)
  while t <= t_it:
    sJnew=RK4(sJ, ts)
    sJ=sJnew
    t=t+ts
    if t>t_store:
      sList.append(sJ)
  return sList

def RK4(sJ, ts):
  k1 = ts*SJdyn(sJ)
  k2 = ts*SJdyn(sJ + k1/2)
  k3 = ts*SJdyn(sJ + k2/2)
  k4 = ts*SJdyn(sJ + k3)
  return sJ + k1/6 + k2/3 + k3/3 + k4/6
  
def GSR(v,n):
   norms=np.zeros(n)
   vnew=v.copy()
   norms[0]=np.linalg.norm(v[:,0])
   vnew[:,0]/=norms[0]
   for j in xrange(1,n):
      for k in xrange(0,j):
         gsp=np.dot(vnew[:,j],vnew[:,k])      #proj.
         vnew[:,j]-=gsp*vnew[:,k]
      norms[j]=np.linalg.norm(vnew[:,j])             #new vect.
      vnew[:,j]/= norms[j]   #norm
   return vnew,norms

def get_traj():
  n=3
  ts = 0.01
  t_max=5000.
  t_store=t_max-4500
  sJ, sJnew = np.zeros(n*(n+1)), np.zeros(n*(n+1))
  sJ[1]=1
  #Generate transient
  sJ = iterate(sJ,t_max, ts, t_store)  
  return sJ[1:]

def lyap_cal():
  n=3
  ts = 0.01
  t_init = 10.
  t_store=t_init-2*ts
  t_max=2000.

  sJ, sJnew = np.zeros(n*(n+1)), np.zeros(n*(n+1))
  sJ[1]=1
  ltot = np.zeros(n)
  traj=[]
  traj_store=t_max-10.
  dyn_steps=10
  print_rate= 5000  
  #Generate transient
  sJ = iterate(sJ,t_init, ts, t_store)[-1]  
  sJ[3],sJ[7],sJ[11]=1,1,1 
  print "init", sJ 
  t=0
  while (t < t_max):
     for j in xrange(dyn_steps): #Dyn. steps
        sJ = RK4(sJ, ts)
        t = t + ts
        if t>traj_store: traj.append(sJ[0:3])
     #GSR 
     vec=sJ.reshape([4,3])[1:4,:]
     nvec,norms=GSR(vec,n)
     sJ[3:12]=nvec.ravel()
     for k in range(n): ltot[k] += np.log(norms[k])          
  return traj,ltot/t

#traj, LE=lyap_cal()
traj=get_traj()
#print "LEs:", LE
np.savetxt("Lorenz.traj", traj)
