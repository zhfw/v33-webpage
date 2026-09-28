import streamlit as st
import pandas as pd
from pathlib import Path
from predictor_core import predict, clean_protein, clean_dna

st.set_page_config(page_title='CFPS Manufacturability Predictor v3.3',page_icon='🧬',layout='wide')
st.markdown('''<style>
.block-container{padding-top:1.4rem;max-width:1320px}.metric-card{padding:14px;border:1px solid #e6e8eb;border-radius:12px;background:#fafbfc}.small{font-size:.86rem;color:#667085}
</style>''',unsafe_allow_html=True)

DEMO_PROTEIN=(Path(__file__).resolve().parent/'data'/'demo_protein.txt').read_text().strip()
DEMO_DNA=(Path(__file__).resolve().parent/'data'/'demo_dna.txt').read_text().strip()
# exact demo DNA is optional; leaving it blank avoids implying a matched coding construct when users edit the protein.

st.title('CFPS Manufacturability Predictor v3.3')
st.caption('First CPU web prototype • trained on uploaded public RP3 / eSOL / solubility / NESG datasets • federated-ready')

tab1,tab2,tab3=st.tabs(['Predictor','Construct diagnostics','Federated network'])
with tab1:
    left,right=st.columns([1.15,0.85])
    with left:
        protein=st.text_area('Protein amino-acid sequence',value=DEMO_PROTEIN,height=220,help='One-letter amino-acid sequence. FASTA header is not needed.')
        dna=st.text_area('Coding DNA (optional)',value=DEMO_DNA,height=120,help='The built-in first run uses the matched coding DNA from the uploaded RP3 test construct.')
        c1,c2,c3=st.columns(3)
        system=c1.selectbox('Expression context',['E. coli CFPS / PURE-like','E. coli recombinant production'])
        temp=c2.selectbox('Planned temperature',['Not specified','16 °C','20 °C','25 °C','30 °C','37 °C'])
        chap=c3.selectbox('Chaperone condition',['Not specified','None','Trigger Factor','GroEL/ES','DnaK/J/GrpE'])
        run=st.button('Run v3.3 prediction',type='primary',use_container_width=True)
    if run:
        try:
            r=predict(protein,dna or None); st.session_state['result']=r
        except Exception as e: st.error(str(e))
    r=st.session_state.get('result')
    with right:
        if r:
            st.subheader('Prediction summary')
            a,b=st.columns(2); a.metric('Overall prioritization',f"{100*r['overall_score']:.0f}/100"); b.metric('Confidence',r['confidence'])
            a,b=st.columns(2); a.metric('CFPS soluble fraction',f"{100*r['cfps_soluble_fraction']:.1f}%"); b.metric('Aggregation / insoluble',f"{100*r['cfps_aggregation_fraction']:.1f}%")
            a,b=st.columns(2); a.metric('RP3 production',f"{100*r['rp3_production_probability_protein']:.1f}%"); b.metric('General solubility',f"{100*r['general_solubility_probability']:.1f}%")
            if r['rp3_production_probability_with_dna'] is not None: st.metric('RP3 production — protein + DNA',f"{100*r['rp3_production_probability_with_dna']:.1f}%")
            a,b=st.columns(2); a.metric('NESG expression score',f"{r['nesg_expression_score_0_5']:.2f}/5"); b.metric('NESG solubility score',f"{r['nesg_solubility_score_0_5']:.2f}/5")
            st.caption(f"OOD index: {r['ood_index']:.2f}. Lower is closer to the public training-feature distribution.")
        else: st.info('Press **Run v3.3 prediction**. A demonstration protein is already loaded.')

    if r:
        st.subheader('Major risks & recommended fixes')
        for rr in r['risks']:
            icon={'high':'🔴','medium':'🟠','low':'🟢'}.get(rr['level'],'•'); st.markdown(f"**{icon} {rr['title']}** — {rr['detail']}")
        for x in dict.fromkeys(r['recommendations']): st.markdown(f"- {x}")
        with st.expander('Model limitations'):
            for x in r['limitations']: st.write('• '+x)

with tab2:
    r=st.session_state.get('result')
    if not r: st.info('Run a prediction first.')
    else:
        f=r['protein_features']
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Length',int(f['length'])); c2.metric('Approx. MW',f"{f['mw_approx']/1000:.1f} kDa"); c3.metric('GRAVY',f"{f['gravy']:.3f}"); c4.metric('Cysteines',int(f['cys_count']))
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Hydrophobic fraction',f"{f['hydrophobic_frac']:.1%}"); c2.metric('Longest hydro. run',int(f['longest_hydrophobic_run'])); c3.metric('Charge proxy',f"{f['charge_proxy']:+.3f}"); c4.metric('Entropy',f"{f['entropy']:.2f}")
        with st.expander('All 42 engineered protein features'):
            st.dataframe(pd.DataFrame({'feature':list(f.keys()),'value':list(f.values())}),use_container_width=True,hide_index=True)
        if r['dna_features']:
            d=r['dna_features']; st.subheader('DNA diagnostics'); c1,c2,c3=st.columns(3); c1.metric('GC',f"{d['gc']:.1%}");c2.metric('GC3',f"{d['gc3']:.1%}");c3.metric('Longest homopolymer',int(d['longest_homopolymer']))

with tab3:
    st.subheader('Three-party CPU federated mode')
    st.markdown('''Each site keeps raw protein/DNA and cached RP3Net-STP features local. Only the small downstream head is federated.''')
    st.code('''Party A: private sequence → frozen RP3Net → 256-D STP → local head ┐\nParty B: private sequence → frozen RP3Net → 256-D STP → local head ├─ FedAvg / FedProx → global v3.3 head\nParty C: private sequence → frozen RP3Net → 256-D STP → local head ┘''')
    m1,m2,m3,m4=st.columns(4); m1.metric('RP3 FedAvg AUROC','0.773');m2.metric('RP3 AUPRC','0.396');m3.metric('CFPS FedProx R²','0.402');m4.metric('CFPS MAE','0.190')
    st.caption('These are the validated engineered-feature CPU baselines before adding the 256-D STP representation.')
    st.markdown('**Federated privacy boundary**: raw protein sequence, DNA sequence, experimental records, and STP feature databases stay at each site. Model-head updates are the federated object.')
