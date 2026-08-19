from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .derivatives import finite_difference_gradient, gn_curvature

@dataclass
class FitResult:
    theta: np.ndarray
    objective: float
    gradient_norm: float
    n_polish_steps: int

def _clip(x,bounds):
    lo=np.array([b[0] for b in bounds]); hi=np.array([b[1] for b in bounds])
    return np.clip(x,lo,hi)

def fit_map(objective,bounds,rng,trial_func,prior_precision,n_starts=5):
    results=[]
    for _ in range(n_starts):
        x0=np.array([rng.uniform(lo,hi) for lo,hi in bounds])
        r=minimize(objective,x0,method="L-BFGS-B",bounds=bounds,
                   options={"maxiter":1000,"gtol":1e-8,"ftol":1e-12})
        results.append(r)
    best=min(results,key=lambda r:r.fun if np.isfinite(r.fun) else np.inf)
    theta=best.x.copy()
    fcur=float(best.fun)
    nsteps=0

    for _ in range(30):
        H=gn_curvature(theta,trial_func,prior_precision)
        g=finite_difference_gradient(objective,theta,1e-6)
        try:
            dx=np.linalg.solve(H,g)
            if (not np.isfinite(dx).all()) or float(g@dx)<=0:
                dx=g.copy()
        except np.linalg.LinAlgError:
            dx=g.copy()

        step=1.0
        improved=False
        while step>=2**-20:
            cand=_clip(theta-step*dx,bounds)
            fnew=float(objective(cand))
            if np.isfinite(fnew) and fnew<fcur:
                prev=fcur
                theta=cand
                fcur=fnew
                nsteps+=1
                improved=True
                break
            step*=0.5
        if not improved:
            break
        if abs(prev-fcur)/(1+abs(fcur))<1e-8:
            break

    g=finite_difference_gradient(objective,theta,1e-6)
    return FitResult(theta,float(fcur),float(np.linalg.norm(g)),nsteps)
