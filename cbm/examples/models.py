"""Three minimal cognitive-model examples in NumPy and JAX."""
from __future__ import annotations
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None


def _softmax_np(x):
    z=x-np.max(x); e=np.exp(z); return e/e.sum()

def _softmax_jax(x):
    z=x-jnp.max(x); e=jnp.exp(z); return e/e.sum()


# ---------- binary RW ----------
def binary_rw_trials(theta, data):
    alpha,beta=theta
    q=np.zeros(2); ll=[]
    for a,r in zip(data["choice"],data["reward"]):
        p=_softmax_np(beta*q)
        ll.append(np.log(np.clip(p[a],1e-12,1)))
        q[a]+=alpha*(r-q[a])
    return np.asarray(ll)

def binary_rw(theta,data):
    return float(np.sum(binary_rw_trials(theta,data)))

def binary_rw_jax(theta,data):
    alpha,beta=theta
    choice=jnp.asarray(data["choice"],dtype=jnp.int32)
    reward=jnp.asarray(data["reward"])
    def step(q,xs):
        a,r=xs
        p=_softmax_jax(beta*q)
        ll=jnp.log(jnp.clip(p[a],1e-12,1))
        q=q.at[a].add(alpha*(r-q[a]))
        return q,ll
    _,ll=jax.lax.scan(step,jnp.zeros(2),(choice,reward))
    return jnp.sum(ll)


# ---------- categorical RW ----------
def categorical_rw_trials(theta,data):
    alpha,beta=theta
    q=np.zeros(3); ll=[]
    for a,r in zip(data["choice"],data["reward"]):
        p=_softmax_np(beta*q)
        ll.append(np.log(np.clip(p[a],1e-12,1)))
        q[a]+=alpha*(r-q[a])
    return np.asarray(ll)

def categorical_rw(theta,data):
    return float(np.sum(categorical_rw_trials(theta,data)))

def categorical_rw_jax(theta,data):
    alpha,beta=theta
    choice=jnp.asarray(data["choice"],dtype=jnp.int32)
    reward=jnp.asarray(data["reward"])
    def step(q,xs):
        a,r=xs
        p=_softmax_jax(beta*q)
        ll=jnp.log(jnp.clip(p[a],1e-12,1))
        q=q.at[a].add(alpha*(r-q[a]))
        return q,ll
    _,ll=jax.lax.scan(step,jnp.zeros(3),(choice,reward))
    return jnp.sum(ll)


# ---------- CES ----------
def ces_trials(theta,data):
    alpha,rho=theta
    x1,x2,y,sigma=data["x1"],data["x2"],data["y"],data["sigma"]
    v=(alpha*x1**rho+(1-alpha)*x2**rho)**(1/rho)
    return -0.5*((y-v)/sigma)**2-np.log(sigma*np.sqrt(2*np.pi))

def ces(theta,data):
    return float(np.sum(ces_trials(theta,data)))

def ces_jax(theta,data):
    alpha,rho=theta
    x1,x2=jnp.asarray(data["x1"]),jnp.asarray(data["x2"])
    y,sigma=jnp.asarray(data["y"]),data["sigma"]
    v=(alpha*x1**rho+(1-alpha)*x2**rho)**(1/rho)
    return jnp.sum(-0.5*((y-v)/sigma)**2-jnp.log(sigma*jnp.sqrt(2*jnp.pi)))
