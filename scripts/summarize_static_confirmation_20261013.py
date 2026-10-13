"""Summarize all frozen G7i contrasts; does not choose models or tune thresholds."""
import argparse,pathlib,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args();summary={'scope':'new static scenes; frozen supervised models; nearest-reference metrics not rendered/control success','domains':{}}
for domain in ['id','ood']:
 r=json.loads((a.root/(domain+'_evaluation/results.json')).read_text());audit=json.loads((a.root/(domain+'_evaluation/independent_audit.json')).read_text());dataaudit=json.loads((a.root/(domain+'_independent_audit/validation.json')).read_text())
 assert dataaudit['status']=='PASS_SAVED_DATA_AUDIT'
 assert audit['status']=='PASS'
 rows=[]
 for n,m in zip(r['models'],r['scene_means']):
  single=[v[1] for v in m[:3]];joint=[v[1] for v in m[3:]]
  rows.append({'model':n,'single_macro_error':sum(v[0] for v in single)/3,'color_target_recovery':single[1][1],'single_macro_nontarget_preservation':sum(v[2] for v in single)/3,'joint_macro_error':sum(v[0] for v in joint)/3,'joint_macro_target_recovery':sum(v[1] for v in joint)/3,'joint_macro_nontarget_preservation':sum(v[2] for v in joint)/3})
 c=r['contrasts']['orth_lambda1'];ci=c['primary_ci98_75'];summary['domains'][domain]={'rows':rows,'primary_names':c['primary_names'],'difference_oblique_minus_orth':c['primary_difference'],'ci98_75':ci,'error_interval_below_zero':ci[1][0]<0,'color_recovery_interval_above_zero':ci[0][1]>0,'data_audit':dataaudit['status'],'independent_evaluation_audit':audit,'scene_count':r['scene_count'],'independent_pairs':r['independent_pairs']}
summary['all_four_predeclared_intervals_in_improving_direction']=all(v['error_interval_below_zero'] and v['color_recovery_interval_above_zero'] for v in summary['domains'].values())
out=a.root/'frozen_summary.json';assert not out.exists();out.write_text(json.dumps(summary,indent=2));print(json.dumps({k:{q:v[q] for q in ['rows','difference_oblique_minus_orth','ci98_75']} for k,v in summary['domains'].items()},indent=2))
