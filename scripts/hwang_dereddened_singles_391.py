"""Deredden Hwang et al. (2024) components (Combined19 x 0.884, zero-point-corrected 1/plx as on our data side) and write single-star (RUWE<1.2 + Halbwachs IPD) dwarf and clump npz files for scripts/calibrate_mist_lowmass_391.py --hwang-dereddened. Usage: hwang_dereddened_singles_391.py OUT_DWARFS.npz OUT_CLUMP.npz"""
import sys, hashlib, yaml, numpy as np
from pathlib import Path
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config
from darkhunter_pop.giants import classify_parent, cmd_for_rows, load_giants_config, parent_row_cmd, classify_evolved
D=Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop/data"); mc=yaml.safe_load(open("config/population/mass_checks_391.yaml"))
for k in ("hwang2024","hwang2024_gaiadr3"): assert hashlib.sha256((D/mc[k]["table"]).read_bytes()).hexdigest()==mc[k]["sha256"]
cfg=load_config(); gc=load_giants_config(Path("config/population/giants.yaml"))
prop=ps.load_proposal_set_fragment(Path("config/population/proposal_set_restart_gen26.yaml")).proposal
parent=ps.load_parent_snapshot(D/"dr3/gaia_snapshots/20261002T234408Z_gaia_source_parent_K1000000_plx0p2_bj",cfg,prop)
pc,ridge=classify_parent(parent,cfg,gc,cmd=parent_row_cmd(parent,cfg,gc))
pcol=parent.columns; pg=np.asarray(pcol["phot_g_mean_mag"],float); snr=np.asarray(pcol["parallax"],float)/np.asarray(pcol["parallax_error"],float)
dzp=parent.truth_parallax_mas-np.asarray(pcol["parallax"],float); ge=np.array([0,11,13,15,17,25.0])
zp=np.array([np.median(dzp[parent.usable&(pg>=a)&(pg<b)&(snr>20)]) for a,b in zip(ge[:-1],ge[1:])])  # same convention as the real side (refit5 extend)
h=Table.read(D/mc["hwang2024"]["table"]); x=Table.read(D/mc["hwang2024_gaiadr3"]["table"],format="ascii.ecsv")
sid=np.concatenate([np.asarray(h["source_id1"],np.int64),np.asarray(h["source_id2"],np.int64)]); mass=np.concatenate([np.asarray(h["predict_mass1_median"],float),np.asarray(h["predict_mass2_median"],float)])
xs=np.asarray(x["Source"],np.int64); o=np.argsort(xs); pos=np.clip(np.searchsorted(xs,sid,sorter=o),0,xs.size-1); j=o[pos]; hit=xs[j]==sid
f=lambda n: np.asarray(x[n],float)[j]
g,br,plx,ep=f("Gmag"),f("BP-RP"),f("Plx")+zp[np.clip(np.digitize(f("Gmag"),ge)-1,0,4)],f("e_Plx")
with np.errstate(divide="ignore",invalid="ignore"):
    rr=np.where(plx>0,1000/plx,np.nan); rlo=np.where(plx>0,1000/(plx+ep),np.nan); rhi=np.where(plx-ep>0,1000/(plx-ep),np.nan)
gal=SkyCoord(ra=f("RA_ICRS")*u.deg,dec=f("DE_ICRS")*u.deg).galactic
c=cmd_for_rows(g,br,gal.l.deg,gal.b.deg,rr,rlo,rhi,cfg,gc)
cut=prop.halbwachs_cuts
single=hit&(f("RUWE")<mc["hwang2024_gaiadr3"]["ruwe_max"])&(f("IPDfmp")<=cut.ipd_frac_multi_peak_max)&(f("IPDgofha")<cut.ipd_gof_harmonic_amplitude_max)&np.isfinite(c.mg0)&np.isfinite(c.colour0)&np.isfinite(mass)
ev=classify_evolved(c.mg0,c.colour0,c.sigma_mu,ridge,gc.provisional_n_sigma); dwarf=ev.classified&~ev.evolved
print(f"components {sid.size}, matched {hit.sum()}, singles (RUWE<1.2 + Halbwachs IPD) {single.sum()}, of which CMD dwarfs {np.sum(single&dwarf)}; median E(B-V)x0.884 {np.nanmedian(c.ebv[single]):.3f} (90th {np.nanpercentile(c.ebv[single],90):.3f}); median d {np.nanmedian(rr[single]):.0f} pc")
s=single&dwarf
np.savez(sys.argv[1],c0=c.colour0[s],mg0=c.mg0[s],mass=mass[s],sigma_mu=c.sigma_mu[s],ebv=c.ebv[s],a_g=c.a_g[s],e_bp_rp=(br-c.colour0)[s])
cl=single&(c.colour0>=1.0)&(c.colour0<1.2)&(c.mg0>=0.3)&(c.mg0<1.3); np.savez(sys.argv[2],c0=c.colour0[cl],mg0=c.mg0[cl],mass=mass[cl],sigma_mu=c.sigma_mu[cl],ebv=c.ebv[cl],a_g=c.a_g[cl],e_bp_rp=(br-c.colour0)[cl])
print(f"dereddened clump box singles: n {cl.sum()}, Hwang median mass {np.nanmedian(mass[cl]):.2f}")
