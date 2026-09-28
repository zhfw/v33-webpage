from __future__ import annotations
import math, json, os
from pathlib import Path
from collections import Counter
import numpy as np
import xgboost as xgb
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except Exception:
    ProteinAnalysis = None

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = set("AVILMFWY")
AA_MASS = {
    'A':89.0935,'C':121.159,'D':133.1032,'E':147.1299,'F':165.19,'G':75.0669,'H':155.1552,'I':131.1736,'K':146.1882,
    'L':131.1736,'M':149.2124,'N':132.1184,'P':115.131,'Q':146.1451,'R':174.2017,'S':105.093,'T':119.1197,'V':117.1469,'W':204.2262,'Y':181.1894
}
KD = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
CODONS = [a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']
DINUCS = [a+b for a in 'ACGT' for b in 'ACGT']
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'

PROTEIN_FEATURE_NAMES = [
 'length','log_length','mw_approx','gravy','charge_proxy','aromatic_frac','hydrophobic_frac','positive_frac','negative_frac','charged_frac','polar_frac','small_frac',
 'cys_count','cys_frac','odd_cys','gly_frac','pro_frac','glypro_frac','unknown_frac','entropy','max_aa_frac','longest_hydrophobic_run'] + [f'frac_{a}' for a in AA_ORDER]
DNA_FEATURE_NAMES = ['dna_length','dna_log_length','gc','gc1','gc2','gc3','n_frac','start_atg','terminal_stop','codon_entropy','max_codon_frac','first60_gc','last60_gc','longest_homopolymer'] + [f'codon_{c}' for c in CODONS] + [f'dinuc_{d}' for d in DINUCS]

def clean_protein(seq:str)->str:
    return ''.join(seq.split()).upper().replace('*','')

def clean_dna(seq:str)->str:
    return ''.join(seq.split()).upper().replace('U','T')

def protein_features(seq:str):
    s=clean_protein(seq)
    if not s: raise ValueError('Protein sequence is empty.')
    L=len(s); c=Counter(s)
    valid=''.join(a if a in AA_ORDER else 'G' for a in s)
    if ProteinAnalysis:
        pa=ProteinAnalysis(valid)
        mw=float(pa.molecular_weight()); gravy=float(pa.gravy())
    else:
        # peptide mass approximation; adequate fallback for model input
        mw=sum(AA_MASS.get(a,75.0669) for a in valid) - 18.01528*(max(0,L-1))
        gravy=sum(KD.get(a,0.0) for a in valid)/L
    cur=mx=0
    for a in s:
        if a in HYDRO: cur+=1; mx=max(mx,cur)
        else: cur=0
    entropy=-sum((n/L)*math.log2(n/L) for n in c.values() if n)
    unknown=sum(n for a,n in c.items() if a not in AA_ORDER)/L
    x=[
        L,math.log1p(L),mw,gravy,(c['K']+c['R']+0.1*c['H']-c['D']-c['E'])/L,
        sum(c[a] for a in 'FWY')/L,sum(c[a] for a in 'AVILMFWY')/L,sum(c[a] for a in 'KRH')/L,
        sum(c[a] for a in 'DE')/L,sum(c[a] for a in 'KRHDE')/L,sum(c[a] for a in 'STNQC')/L,
        sum(c[a] for a in 'ACGSTV')/L,float(c['C']),c['C']/L,float(c['C']%2),c['G']/L,c['P']/L,
        (c['G']+c['P'])/L,unknown,entropy,max(c.values())/L,float(mx)
    ] + [c[a]/L for a in AA_ORDER]
    return np.asarray(x,dtype=np.float32), dict(zip(PROTEIN_FEATURE_NAMES,x))

def _gc(s):
    if not s: return 0.0
    return (s.count('G')+s.count('C'))/len(s)

def dna_features(seq:str):
    s=clean_dna(seq)
    if not s: raise ValueError('DNA sequence is empty.')
    L=len(s); c=Counter(s)
    frame=[s[i:i+3] for i in range(0,L-2,3)]
    valid_codons=[x for x in frame if len(x)==3 and set(x)<=set('ACGT')]
    cc=Counter(valid_codons); nc=max(1,len(valid_codons))
    cent=-sum((n/nc)*math.log2(n/nc) for n in cc.values() if n)
    maxcf=max(cc.values())/nc if cc else 0.0
    gcpos=[]
    for offset in range(3):
        chars=[s[i] for i in range(offset,L,3) if s[i] in 'ACGT']
        gcpos.append(_gc(''.join(chars)))
    longest=cur=0; prev=None
    for a in s:
        if a==prev: cur+=1
        else: prev=a; cur=1
        longest=max(longest,cur)
    x=[L,math.log1p(L),_gc(''.join(a for a in s if a in 'ACGT')),gcpos[0],gcpos[1],gcpos[2],
       sum(1 for a in s if a not in 'ACGT')/L,float(s.startswith('ATG')),float(s[-3:] in {'TAA','TAG','TGA'}),cent,maxcf,
       _gc(s[:60]),_gc(s[-60:]),float(longest)]
    x += [cc[codon]/nc for codon in CODONS]
    dc=Counter(s[i:i+2] for i in range(L-1) if set(s[i:i+2])<=set('ACGT')); nd=max(1,sum(dc.values()))
    x += [dc[d]/nd for d in DINUCS]
    return np.asarray(x,dtype=np.float32), dict(zip(DNA_FEATURE_NAMES,x))

def _load(name):
    b=xgb.Booster(); b.load_model(str(MODEL_DIR/name)); return b

def _pred(model,x): return float(model.predict(xgb.DMatrix(np.asarray(x,dtype=np.float32).reshape(1,-1)))[0])

_MODELS=None
def models():
    global _MODELS
    if _MODELS is None:
        _MODELS={
          'general':[_load(f'general_solubility_fold{i}_xgb_fast.json') for i in range(5)],
          'cfps':[_load(f'cfps_esol_fold{i}_xgb_fast.json') for i in range(5)],
          'rp3p':_load('rp3_protein_final_xgb.json'),
          'rp3dna':_load('rp3_dna_protein_plus_dna_final_xgb.json'),
          'nesg_exp':_load('nesg_exp_final_xgb.json'),
          'nesg_sol':_load('nesg_sol_final_xgb.json'),
        }
    return _MODELS

def risk_assessment(f,dna_f=None):
    risks=[]; rec=[]
    def add(level,title,detail,fix): risks.append({'level':level,'title':title,'detail':detail}); rec.append(fix)
    if f['hydrophobic_frac']>0.48 or f['longest_hydrophobic_run']>=8: add('high','Hydrophobic aggregation risk',f"Hydrophobic fraction {f['hydrophobic_frac']:.2f}; longest hydrophobic run {int(f['longest_hydrophobic_run'])} aa.",'Consider domain trimming, surface-polar substitutions outside functional sites, lower-temperature expression, or chaperone support.')
    if f['odd_cys']>0: add('medium','Odd cysteine count',f"{int(f['cys_count'])} cysteines may increase disulfide/mispairing sensitivity.",'Check intended disulfide pattern; for reducing CFPS consider cysteine redesign only when structurally permissible.')
    if f['cys_frac']>0.04: add('medium','High cysteine burden',f"Cysteine fraction {f['cys_frac']:.1%}.",'Use an oxidative/disulfide-supporting CFPS condition if disulfides are required.')
    if f['length']>800: add('medium','Long construct',f"Length {int(f['length'])} aa is outside the dense region of the training corpus.",'Consider domain constructs or validate with a matched long-protein control.')
    if abs(f['charge_proxy'])>0.12: add('medium','Extreme net-charge proxy',f"Charge proxy {f['charge_proxy']:+.3f}.",'Review strongly charged termini/patches and buffer/salt conditions.')
    if f['entropy']<3.5 or f['max_aa_frac']>0.20: add('medium','Low-complexity bias',f"Sequence entropy {f['entropy']:.2f}; max single-residue fraction {f['max_aa_frac']:.1%}.",'Consider removing nonessential low-complexity tails or testing shorter constructs.')
    if dna_f:
        if dna_f['gc']<0.35 or dna_f['gc']>0.70: add('medium','Extreme coding GC',f"Coding GC {dna_f['gc']:.1%}.",'Synonymously redesign toward a moderate host-appropriate GC range while preserving amino-acid sequence.')
        if dna_f['longest_homopolymer']>=7: add('medium','Long DNA homopolymer',f"Longest homopolymer {int(dna_f['longest_homopolymer'])} nt.",'Break long homopolymers by synonymous codon changes.')
        if dna_f['n_frac']>0: add('high','Ambiguous DNA bases',f"Ambiguous-base fraction {dna_f['n_frac']:.2%}.",'Resolve ambiguous DNA bases before synthesis or model-based codon analysis.')
    if not risks:
        risks=[{'level':'low','title':'No major sequence-level red flag','detail':'The first-pass feature rules did not detect a dominant risk.'}]
        rec.append('Proceed to system-specific validation; use the model probabilities as prioritization, not a guarantee of yield.')
    return risks,rec

def predict(protein_seq:str,dna_seq:str|None=None):
    x,f=protein_features(protein_seq); m=models()
    p_general=float(np.mean([_pred(mm,x) for mm in m['general']]))
    p_cfps=float(np.clip(np.mean([_pred(mm,x) for mm in m['cfps']]),0,1))
    p_prod=_pred(m['rp3p'],x)
    exp=float(np.clip(_pred(m['nesg_exp'],x),0,5)); sol=float(np.clip(_pred(m['nesg_sol'],x),0,5))
    df=None; p_prod_dna=None
    if dna_seq and clean_dna(dna_seq):
        dx,df=dna_features(dna_seq); p_prod_dna=_pred(m['rp3dna'],np.concatenate([x,dx]))
    prod=p_prod_dna if p_prod_dna is not None else p_prod
    overall=float(np.clip(0.35*prod+0.35*p_cfps+0.20*p_general+0.10*(exp/5),0,1))
    scaler=np.load(MODEL_DIR/'public_engineered_scaler.npz',allow_pickle=True); z=np.abs((x-scaler['mean'])/scaler['std']); ood=float(np.mean(np.minimum(z,10)))
    confidence='higher' if ood<1.5 else ('moderate' if ood<2.5 else 'lower / OOD')
    risks,recs=risk_assessment(f,df)
    return {
      'overall_score':overall,'general_solubility_probability':p_general,'cfps_soluble_fraction':p_cfps,'cfps_aggregation_fraction':1-p_cfps,
      'rp3_production_probability_protein':p_prod,'rp3_production_probability_with_dna':p_prod_dna,'nesg_expression_score_0_5':exp,'nesg_solubility_score_0_5':sol,
      'confidence':confidence,'ood_index':ood,'protein_features':f,'dna_features':df,'risks':risks,'recommendations':recs,
      'limitations':['Absolute CFPS yield (mg/mL) is not trained in this first web release.','CHO secretion/yield is not yet model-backed in this release.','Aggregation is operationally defined as 1 - eSOL/PURE soluble fraction.','The overall score is a composite prioritization score, not a directly measured endpoint.']
    }

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--protein',default=''); ap.add_argument('--protein-file',default=''); ap.add_argument('--dna',default=''); ap.add_argument('--dna-file',default=''); a=ap.parse_args()
    p=a.protein or (Path(a.protein_file).read_text() if a.protein_file else '')
    d=a.dna or (Path(a.dna_file).read_text() if a.dna_file else '')
    print(json.dumps(predict(p,d or None),indent=2))
