"""Single-thread CPU time of one adsorption step (256 cells): Paper 1 solver_fd.py against verify/consistent_column.py.
Run with OMP_NUM_THREADS=1. Prints CPU seconds and RHS evaluation counts."""
import sys,time,numpy as np
import os
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE); sys.path.insert(0,os.path.abspath(os.path.join(HERE,"..","..","ai-mof-cof-dynamics")))
from consistent_column import Column, from_paper1
from gen_parametric_dataset import build_physics, rh_to_conc
from solver_fd import generate_breakthrough_data
from scipy.integrate import solve_ivp
p=build_physics(dict(q_max=24.5, delta_H=-52000.0, step_rh=0.12, isotherm_n=3.0, henry_fraction=0.17, d_p=3.25e-3, rho_p=1050.0, eps_t=0.39),dict(T_in=303.15, rh_feed=0.4, v=0.1))
c_in=rh_to_conc(0.4,303.15); t_st=p.stoichiometric_time(c_in,303.15); N=256; T=0.8*t_st
t0=time.process_time(); generate_breakthrough_data(p,N_z=N,t_final=T,c_in=c_in,n_snapshots=201,rtol=1e-6,atol=1e-10,verbose=True); print("paper1 cpu",round(time.process_time()-t0,1),flush=True)
for cw in (0.0,33.6):
  col=Column(p,N,cw,iso=from_paper1(p)); y0=col.initial_state(303.15); inp=(c_in,303.15,303.15)
  t0=time.process_time(); sol=solve_ivp(col.rhs,(0,T),y0,method="BDF",args=(inp,),t_eval=np.linspace(0,T,201),jac_sparsity=col.sparsity(),rtol=1e-6,atol=1e-10)
  print("mine c_w",cw,"cpu",round(time.process_time()-t0,1),"nfev",sol.nfev,"njev",sol.njev,"nlu",sol.nlu,flush=True)
