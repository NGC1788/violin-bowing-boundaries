"""Read verified source arrays and report schema/quality, never ML scores.

Only performance fields and pseudonymous cohort counts are inspected. Personal
questionnaires/demographic values are not exported. Source MATLAB is not run.
"""
from pathlib import Path
import argparse, collections, hashlib, importlib.metadata, json, sys, time
import numpy as np
from download_blanco import digest, NAME, MD5

HERE = Path(__file__).resolve().parent
GS_FIELDS = ('Violin_Trials_GS','Violin_Trials_GS_online','Violin_Trials_GS_online_MC_Task','Violin_Trials_GS_online_GS_Task')
STAGES = ('Results_pre','Results_feed','Results_post')
SOUND = ('f0_raw_data','power_raw_data','ap_raw_data')
MOTION = ('bowSkew_raw_data','tilt_raw_data','bowBridge_raw_data','Speed_raw_data','inclination_raw_data','ViolinDist_raw_data','bowPosition_raw_data')


def structure(value):
    if isinstance(value, np.ndarray) and value.dtype.names and value.size:
        return value.ravel()[0]
    return None


def cells(value):
    a = np.asarray(value)
    if a.size == 0:
        return []
    if a.dtype.kind == 'O':
        return [np.asarray(x,dtype=float).ravel() for x in a.ravel()]
    return [np.asarray(a,dtype=float).ravel()]


def quantiles(values):
    a = np.asarray(values,dtype=float)
    a = a[np.isfinite(a)]
    return dict(zip(('min','p05','median','p95','max'),map(float,np.quantile(a,[0,.05,.5,.95,1])))) if len(a) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path.home()/'violin-rne/data/external/blanco2021')
    args=parser.parse_args();root=args.data_root.expanduser().resolve()
    sys.path.insert(0,str(root/'python_packages'))
    import matio
    assert digest(root/NAME,'md5') == MD5
    mat = root/'extracted/Sound_and_Motion_Data/participants_raw_Sound_Motion.mat'
    start=time.perf_counter();data=matio.load_from_mat(mat,raw_data=True)
    load_seconds=time.perf_counter()-start
    counts=collections.Counter();participant_counts={};schemas={};signals={};pairs=[];aggregate_checks=[]
    required_mismatch=0;empty_sound_trials=0;no_explicit_timestamp_fields=True
    missing_motion_fields=collections.Counter();motion_schema_variants=collections.Counter()
    alignment_counts=collections.Counter();aggregate_lengths=collections.Counter()
    content_owners=collections.defaultdict(list);within_mode_duplicate_positions=collections.Counter()
    for name in SOUND+MOTION:
        signals[name]={'trials':0,'samples':0,'nonfinite':0,'empty':0,'negative':0,
                       'adjacent_equal':0,'adjacent_pairs':0,'lengths':[],'sampled_values':[]}
    for group in ('CG','FG','EG'):
        people=data['participants_'+group];participant_counts[group]=people.size
        for person_index,person in enumerate(people.ravel()):
            for stage in STAGES:
                results=person.properties[stage]
                if not hasattr(results,'properties'):continue
                for field in GS_FIELDS:
                    gs=structure(results.properties[field])
                    if gs is None:continue
                    no_explicit_timestamp_fields &= not any('time' in x.lower() for x in gs.dtype.names)
                    schemas['sound_fields']=list(gs.dtype.names)
                    arrays={name:cells(gs[name]) for name in SOUND}
                    lengths=[len(x) for x in arrays.values()]
                    if len(set(lengths))!=1:raise AssertionError('Sound channel trial count differs')
                    count=lengths[0]
                    counts[f'{group}/{stage}/{field}']+=count
                    for aggregate in ('pitch_stab','dyn_stab'):
                        na=np.asarray(gs[aggregate]).size
                        aggregate_lengths[f'{group}/{stage}/{field}/{aggregate}']+=na
                        alignment_counts['sound_raw_vs_'+aggregate+'_cell_count_mismatch']+=int(count!=na)
                    motion_field=field.replace('_GS','_MC',1)
                    mc=structure(results.properties[motion_field])
                    motion_arrays={}
                    if mc is not None:
                        schemas['motion_fields']=list(mc.dtype.names)
                        motion_schema_variants['|'.join(mc.dtype.names)]+=1
                        no_explicit_timestamp_fields &= not any('time' in x.lower() for x in mc.dtype.names)
                        motion_arrays={name:cells(mc[name]) if name in mc.dtype.names else [] for name in MOTION}
                        for name in MOTION:
                            missing_motion_fields[name]+=int(name not in mc.dtype.names)
                        alignment_counts['sound_raw_vs_motion_cell_count_mismatch']+=int(count!=len(motion_arrays['bowSkew_raw_data']))
                    for name,items in {**arrays,**motion_arrays}.items():
                        stat=signals[name]
                        for a in items:
                            stat['trials']+=1;stat['samples']+=len(a);stat['nonfinite']+=int((~np.isfinite(a)).sum())
                            stat['negative']+=int((a<0).sum());stat['empty']+=int(len(a)==0);stat['lengths'].append(len(a))
                            if len(a)>1:
                                stat['adjacent_equal']+=int(np.sum(a[1:]==a[:-1]));stat['adjacent_pairs']+=len(a)-1
                            if len(a):
                                stat['sampled_values'].extend(a[np.linspace(0,len(a)-1,min(32,len(a)),dtype=int)].tolist())
                    for i in range(count):
                        ns=[len(arrays[name][i]) for name in SOUND]
                        required_mismatch+=int(len(set(ns))!=1);empty_sound_trials+=int(min(ns)==0)
                        h=hashlib.sha256()
                        for channel in SOUND:
                            h.update(arrays[channel][i].astype('<f8').tobytes())
                        owners=content_owners[h.hexdigest()]
                        for previous in owners:
                            if previous[:4]==(group,person_index,stage,field):
                                within_mode_duplicate_positions[f'{stage}/{field}/{previous[4]}={i}']+=1
                        owners.append((group,person_index,stage,field,i))
                        if mc is not None and i<len(motion_arrays['bowSkew_raw_data']):
                            nm=len(motion_arrays['bowSkew_raw_data'][i])
                            if nm and min(ns)>0:
                                pairs.append((ns[0],nm))
                        # Compare source aggregate definitions, not predictions.
                        for raw_name,aggregate_name in [('f0_raw_data','pitch_stab'),('power_raw_data','dyn_stab')]:
                            values=arrays[raw_name][i];stored=np.asarray(gs[aggregate_name],dtype=float).ravel()
                            if i<len(stored) and len(values)>1:
                                finite=values[np.isfinite(values)]
                                if len(finite)>1 and np.isfinite(stored[i]):
                                    sd=float(np.std(finite,ddof=1));reference=float(stored[i])
                                    aggregate_checks.append({'field':raw_name,'raw_std':sd,'stored_std':reference,
                                                             'absolute_error':abs(sd-reference)})
    report={'scope':'Schema and signal-quality audit only; no model fitting, prediction scores or human quality judgement',
            'mat_sha256':digest(mat,'sha256'),'mat_bytes':mat.stat().st_size,'mat_reader':'mat-io',
            'parser_versions':{n:importlib.metadata.version(n) for n in ['mat-io','pandas','h5py','numpy','scipy']},
            'load_seconds':load_seconds,'participant_records_before_exclusions':participant_counts,
            'participant_records_total':sum(participant_counts.values()),'counts_by_cohort_stage_mode':dict(counts),
            'sound_trial_cells_total':sum(counts.values()),'empty_sound_trial_cells':empty_sound_trials,
            'sound_channel_length_mismatches':required_mismatch,'schemas':schemas,
            'missing_motion_fields_by_participant_condition':dict(missing_motion_fields),
            'motion_schema_variants':dict(motion_schema_variants),
            'trial_count_alignment_checks':dict(alignment_counts),
            'aggregate_cell_counts_by_cohort_stage_mode':dict(aggregate_lengths),
            'exact_content_duplicates':{'unique_three_channel_sequences':len(content_owners),
                'duplicate_extra_cells':sum(len(x)-1 for x in content_owners.values()),
                'hashes_repeated_across_participants':sum(len({x[:2] for x in owners})>1 for owners in content_owners.values()),
                'within_mode_duplicate_positions':dict(within_mode_duplicate_positions),
                'scope':'Byte-identical f0,power,aperiodicity triples; do not count duplicated cells as independent trials'},
            'explicit_timestamp_fields_found':not no_explicit_timestamp_fields,'signals':{}}
    for name,s in signals.items():
        report['signals'][name]={k:v for k,v in s.items() if k not in ('lengths','sampled_values')}
        report['signals'][name].update(length_quantiles=quantiles(s['lengths']),sampled_value_quantiles=quantiles(s['sampled_values']),
            sampled_value_quantile_scope='At most32 uniformly indexed samples per trial; descriptive only',
            adjacent_equal_fraction=s['adjacent_equal']/s['adjacent_pairs'] if s['adjacent_pairs'] else None)
    a=np.asarray(pairs,dtype=float)
    report['paired_modal_lengths']={'pairs':len(pairs),'sound_to_motion_length_ratio':quantiles(a[:,0]/a[:,1]),
        'inferred_duration_difference_if_audio_hop32_at44100Hz_and_motion512_at44100Hz':quantiles(a[:,0]*32/44100-a[:,1]*512/44100),
        'warning':'Rates inferred from cited toolbox/default and paper, not saved timestamps. Length agreement does not prove common origin or synchronization.'}
    report['aggregate_consistency']={}
    for name in ('f0_raw_data','power_raw_data'):
        entries=[x for x in aggregate_checks if x['field']==name]
        report['aggregate_consistency'][name]={'trials':len(entries),'absolute_error_quantiles':quantiles([x['absolute_error'] for x in entries]),
            'within_1e_6':sum(x['absolute_error']<1e-6 for x in entries),
            'stored_to_recomputed_ratio':quantiles([x['stored_std']/x['raw_std'] for x in entries if x['raw_std']>0])}
    report['unresolved_before_temporal_protocol']=['Exact acoustic hop/window alignment and recorded parameters',
        'Cross-modal time origin and dropouts; do not resample to equal lengths using future duration',
        'f0 units differ from README Hz label; cited YIN uses octaves relative440Hz',
        'Kinect repeated samples/large gaps; not independent86Hz motion measurements',
        'No acoustic waveform in this archive; separate2.2GB record not downloaded']
    (HERE/'schema_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('signals','counts_by_cohort_stage_mode','schemas')},indent=2))


if __name__=='__main__':main()
