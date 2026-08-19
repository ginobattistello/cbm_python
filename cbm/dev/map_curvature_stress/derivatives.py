from __future__ import annotations
import numpy as np

def negative_log_posterior_np(theta, model, data, prior_mean, prior_precision):
    from .models import trial_loglik_np
    theta = np.asarray(theta, float)
    ll = np.sum(trial_loglik_np(model, theta, data))
    d = theta - np.asarray(prior_mean, float)
    lp = -0.5*d @ np.asarray(prior_precision,float) @ d
    return float(-(ll+lp))

def negative_log_posterior_jax(theta, model, data, prior_mean, prior_precision):
    import jax.numpy as jnp
    from .models import trial_loglik_jax
    ll = jnp.sum(trial_loglik_jax(model, theta, data))
    d = theta - jnp.asarray(prior_mean)
    P = jnp.asarray(prior_precision)
    lp = -0.5*(d @ P @ d)
    return -(ll+lp)

def finite_difference_gradient(fun, x, eps=1e-6):
    x = np.asarray(x,float)
    g = np.zeros_like(x)
    for i in range(x.size):
        h = eps*max(1.0,abs(x[i]))
        xp=x.copy(); xm=x.copy()
        xp[i]+=h; xm[i]-=h
        g[i]=(fun(xp)-fun(xm))/(2*h)
    return g

def original_cbm_hessian(fun, x, epsilon=1e-5):
    from scipy.optimize import approx_fprime
    x=np.asarray(x,float)
    n=x.size
    H=np.zeros((n,n))
    g0=approx_fprime(x,fun,epsilon)
    for i in range(n):
        xp=x.copy()
        xp[i]+=epsilon
        gp=approx_fprime(xp,fun,epsilon)
        H[i,:]=(gp-g0)/epsilon
    return 0.5*(H+H.T)

def central_fd_hessian(fun, x, eps=1e-4):
    x=np.asarray(x,float)
    n=x.size
    H=np.zeros((n,n))
    hs=np.array([eps*max(1.0,abs(v)) for v in x])
    f0=float(fun(x))
    for i in range(n):
        ei=np.zeros(n); ei[i]=hs[i]
        H[i,i]=(fun(x+ei)-2*f0+fun(x-ei))/(hs[i]**2)
        for j in range(i+1,n):
            ej=np.zeros(n); ej[j]=hs[j]
            v=(fun(x+ei+ej)-fun(x+ei-ej)-fun(x-ei+ej)+fun(x-ei-ej))/(4*hs[i]*hs[j])
            H[i,j]=v; H[j,i]=v
    return 0.5*(H+H.T)

def gn_curvature(theta, trial_func, prior_precision, relative_step=1e-4, absolute_floor=1e-4):
    theta=np.asarray(theta,float)
    f0=np.asarray(trial_func(theta),float)
    J=np.zeros((f0.size,theta.size))
    for i in range(theta.size):
        dx=relative_step*theta[i]
        if abs(dx)<=absolute_floor:
            dx=absolute_floor
        xp=theta.copy(); xp[i]+=dx
        J[:,i]=(np.asarray(trial_func(xp),float)-f0)/dx
    H=J.T@J + np.asarray(prior_precision,float)
    return 0.5*(H+H.T)

def autodiff_hessian(theta, model, data, prior_mean, prior_precision):
    import jax, jax.numpy as jnp
    fun=lambda z: negative_log_posterior_jax(z,model,data,prior_mean,prior_precision)
    H=np.asarray(jax.hessian(fun)(jnp.asarray(theta,dtype=jnp.float64)),float)
    return 0.5*(H+H.T)

def relative_frobenius(H,H_ref):
    H=np.asarray(H,float); H_ref=np.asarray(H_ref,float)
    denom=max(np.linalg.norm(H_ref,ord="fro"),np.finfo(float).eps)
    return float(np.linalg.norm(H-H_ref,ord="fro")/denom)

def eig_summary(H):
    H=0.5*(np.asarray(H,float)+np.asarray(H,float).T)
    eig=np.linalg.eigvalsh(H)
    mn=float(eig[0]); mx=float(eig[-1]); pd=bool(mn>0)
    if pd:
        cond=float(mx/mn)
        log10cond=float(np.log10(cond))
        sign,logdet=np.linalg.slogdet(H)
        logdet=float(logdet) if sign>0 else np.nan
    else:
        cond=np.inf; log10cond=np.inf; logdet=np.nan
    return {
        "is_pd":pd,
        "min_eig":mn,
        "max_eig":mx,
        "condition_number":cond,
        "log10_condition":log10cond,
        "logdet":logdet,
    }

def laplace_logevidence(objective_map,H):
    info=eig_summary(H)
    if not info["is_pd"]:
        return np.nan
    d=np.asarray(H).shape[0]
    return float(-objective_map + 0.5*d*np.log(2*np.pi) - 0.5*info["logdet"])
