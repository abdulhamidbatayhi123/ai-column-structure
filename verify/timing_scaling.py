"""CPU time of one temperature-swing cycle at 128/256/512 cells, n = 3 and 6 (single thread).
Writes verify/timing_scaling.json. Used by docs/compute_3B.md, Section 0."""
import sys,time,json,numpy as np
sys.path.insert(0,"C:/Users/abdulhamid batayhi/Desktop/ai-column-structure/verify"); sys.path.insert(0,"C:/Users/abdulhamid batayhi/Desktop/ai-mof-cof-dynamics")
from consistent_column import Column
from gen_parametric_dataset import build_physics, rh_to_conc
from scipy.integrate import solve_ivp
out=[]
for nexp in (3.0,6.0):
  p=build_physics(dict(q_max=24.5, delta_H=-52000.0, step_rh=0.12, isotherm_n=nexp, henry_fraction=0.17, d_p=3.25e-3, rho_p=1050.0, eps_t=0.39),dict(T_in=303.15, rh_feed=0.4, v=0.1))
  c_in=rh_to_conc(0.4,303.15); t_st=p.stoichiometric_time(c_in,303.15)
  for N in (128,256,512):
    col=Column(p,N,33.6); y=col.initial_state(303.15); rec={"n":nexp,"N":N}
    for name,dur,inp in (("ads",0.8*t_st,(c_in,303.15,303.15)),("des",0.6*t_st,(c_in,353.15,353.15))):
      t0=time.process_time(); sol=solve_ivp(col.rhs,(0,dur),y,method="BDF",args=(inp,),t_eval=np.linspace(0,dur,201),jac_sparsity=col.sparsity(),rtol=1e-6,atol=1e-10)
      rec[name+"_cpu"]=round(time.process_time()-t0,2); rec[name+"_nfev"]=int(sol.nfev); y=sol.y[:,-1]
    out.append(rec); print(rec,flush=True)
json.dump(out,open("C:/Users/abdulhamid batayhi/Desktop/ai-column-structure/verify/timing_scaling.json","w"),indent=1)
