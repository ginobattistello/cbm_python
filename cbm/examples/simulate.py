from __future__ import annotations
import numpy as np

def softmax(x):
    z=x-np.max(x); e=np.exp(z); return e/e.sum()

def binary_subject(rng, theta=(0.35,3.0), n=150):
    alpha,beta=theta; q=np.zeros(2); c=[]; r=[]
    p_reward=np.array([0.75,0.25])
    for _ in range(n):
        p=softmax(beta*q); a=rng.choice(2,p=p)
        rew=float(rng.random()<p_reward[a])
        c.append(a); r.append(rew); q[a]+=alpha*(rew-q[a])
    return {"choice":np.asarray(c),"reward":np.asarray(r)}

def categorical_subject(rng, theta=(0.35,3.0), n=150):
    alpha,beta=theta; q=np.zeros(3); c=[]; r=[]
    p_reward=np.array([0.75,0.50,0.25])
    for _ in range(n):
        p=softmax(beta*q); a=rng.choice(3,p=p)
        rew=float(rng.random()<p_reward[a])
        c.append(a); r.append(rew); q[a]+=alpha*(rew-q[a])
    return {"choice":np.asarray(c),"reward":np.asarray(r)}

def ces_subject(rng, theta=(0.60,0.30), n=150, sigma=0.20):
    alpha,rho=theta
    x1=rng.uniform(0.5,2,n); x2=rng.uniform(0.5,2,n)
    v=(alpha*x1**rho+(1-alpha)*x2**rho)**(1/rho)
    y=v+rng.normal(0,sigma,n)
    return {"x1":x1,"x2":x2,"y":y,"sigma":sigma}
