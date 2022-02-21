import pandas as pd
import numpy as np

import os
import argparse
from omegaconf import OmegaConf

from joblib import Parallel, delayed

from core.lib.preprocessing import normalize_data, mean_filter_normalize, my_find_peaks, identify_out_pk_vly, rm_baseline_wander

from core.lib.features_extraction import compute_sp_dp, extract_cycle_check, extract_feat_cycle, extract_feat_original

def _print_step(s,cl_log):
	print("--- {} ---".format(s))
	cl_log.write("--- {} ---\n".format(s))

def _print_n_samples(df, cl_log, sent = "data size: "):
	rec = df.trial.map(lambda s: s[:s.rfind('_')])
	stats = [df.patient.unique().shape[0], rec.unique().shape[0], df.shape[0]]
	print(sent+" {}, {}, {}".format(*stats))
	cl_log.write(sent+" {}, {}, {} \n".format(*stats))

def _filter_ppg(df, args):
	""" Compute and save filter signal in dataframe. """
	if args.ppg_filter.enable:
		df['fsignal']=Parallel( n_jobs=args.parallel.n_jobs, 
								verbose=args.parallel.verbose)(
								delayed(mean_filter_normalize)( sig, fs=args.fs,
																lowcut=args.ppg_filter.lowcut,
																highcut=args.ppg_filter.highcut,
																order=1) for sig in df.signal)
	else:
		df['fsignal']=Parallel( n_jobs=args.parallel.n_jobs, 
								verbose=args.parallel.verbose)(
								delayed(normalize_data)(sig) for sig in df.signal)
	return df

def _extract_c(sig, fs, pk_th=0.6, remove_start_end=True):
	""" Wrapper for extract_cycle_check function. """
	try:
	    cs, pks_norm, flag1, flag2, pks, vlys = extract_cycle_check(sig, fs, pk_th, remove_start_end)
	except:
	    cs, pks_norm, flag1, flag2, pks, vlys = [], [], True, True, [], []
	return cs, pks_norm, flag1, flag2, pks, vlys

def _rm_baseline_wander(ppg, vlys, add_pts = True):
	""" Wrapper for rm_baseline_wander function. """
	return rm_baseline_wander(ppg, vlys, add_pts = True)[0]

def _compute_naive_BP(df):
	df['SP'] = df.abp_signal.map(np.max)
	df['DP'] = df.abp_signal.map(np.min)
	return df

def _compute_quality_idx(df):
	""" Compute the statistics to do the distorted signal removal """

	name_tab = {'pks': 'p2p','vlys': 'v2v'}
	s_name = {'ppg': 'signal','abp': 'abp_signal'}
	for s_type in ['ppg', 'abp']:
		for target in ['pks', 'vlys']:
			pre1 = '{}_{}'.format(s_type, target) #ex: abp_pks
			pre2 = '{}_{}'.format(s_type, name_tab[target]) #ex: abp_p2p

			df[pre2+'_dif'] = df[pre1].map(np.diff) # p2p or v2v dif
			df[pre2] = df[pre2+'_dif'].map(np.median) # p2p or v2v med
			df[pre2+'_std'] = df[pre2+'_dif'].map(np.std) # p2p or v2v std
			df[pre1+'_amp'] = df.apply(lambda row: row[s_name[s_type]][row[pre1]], axis=1) # pks or vlys amp
			df[pre1+'_amp_std']=df[pre1+'_amp'].map(np.std) # pks or vlys amp std
			df[pre1+'_amp_dif_std'] = df[pre1+'_amp'].map(np.diff).map(np.std) # pks or vlys amp dif std

	return df

def _compute_sp_dp(abp, fs, pk_th=0.6, remove_start_end=False):
	""" Wrapper for compute_sp_dp function. """
	try:
		return compute_sp_dp(abp, fs, pk_th, remove_start_end)
	except:
		return -1, -1, True, True, [], []

#---------- Step Functions ----------#
def _abnormal_BP(df, args, cl_log, isABP):
	""" Filter by ABP/BP values outside ranges.
		isABP parameters allows to print the tags when is ABP limits or BP values.
	"""
	up_sbp, lo_sbp = args.bp_filter.up_sbp, args.bp_filter.lo_sbp
	up_dbp, lo_dbp = args.bp_filter.up_dbp, args.bp_filter.lo_dbp
	up_diff, lo_diff = args.bp_filter.up_diff, args.bp_filter.lo_diff

	#Limit BP labels
	df['bp_dif'] = df['SP'] - df['DP']
	BP_max = (df.SP >= lo_sbp) & (df.SP <= up_sbp)
	BP_min = (df.DP >= lo_dbp) & (df.DP <= up_dbp)
	BP_dif = (df.bp_dif >= lo_diff) & (df.bp_dif <= up_diff)

	tag1, tag2, tag3 = ['ABP_max', 'ABP_min', 'ABP_dif'] if isABP else ['SBP', 'DBP', 'BP-dif']
	cl_log.write(" - removed by {} range: {} \n".format(tag1, (~BP_max).sum()))
	cl_log.write(" - removed by {} range: {} \n".format(tag2,(~BP_min).sum()))
	cl_log.write(" - removed by {} range: {} \n".format(tag3, (~BP_dif).sum()))

	df = df[BP_max & BP_min & BP_dif].reset_index(drop=True)

	return df

def _extract_ppg_cycles(df, args, cl_log):
	""" Compute and save the ppg cycles in dataframe.
		This function updates filtered signal. 
	"""
	df = _filter_ppg(df, args)

	c_ppg = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(_extract_c)(sig, fs=args.fs, remove_start_end=args.remove_start_end) for sig in df.fsignal)

	for i, label in enumerate(['cs','pks_norm','ppg_f1','ppg_f2','ppg_pks','ppg_vlys']):
		df[label] = [val[i] for val in c_ppg]

	not_computed = ((df['ppg_pks'].map(len)==0) | (df['ppg_f2']))
	cl_log.write(" - removed by not computed: {} \n".format((not_computed).sum()))
	df = df[~not_computed].reset_index(drop=True)

	return df

def _all_peaks_valleys(df, args, cl_log):
	df['abp_pks'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(my_find_peaks)(sig, args.fs) for sig in df.abp_signal)
	df['abp_vlys'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(my_find_peaks)(-sig, args.fs) for sig in df.abp_signal)
	df['ppg_pks'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(my_find_peaks)(sig, args.fs) for sig in df.signal)
	df['ppg_vlys'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(my_find_peaks)(-sig, args.fs) for sig in df.signal)

	not_computed = ((df['abp_pks'].map(len) < 1) | (df['abp_vlys'].map(len) < 2) | 
                (df['ppg_pks'].map(len) < 1) | (df['ppg_vlys'].map(len) < 2))

	cl_log.write(" - removed by not_computed: {}\n".format((df.patient.isin(df[not_computed].patient)).sum()))
	df=df[~df.patient.isin(df[not_computed].patient)].reset_index(drop=True)

	return df

def _peaks_valleys_outliers(df, args, cl_log):
	th_out = args.th_out
	df['out_abp'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(identify_out_pk_vly)(row.abp_signal, row.abp_pks, row.abp_vlys, th=th_out) for i, row in df.iterrows())
	df['out_ppg'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(identify_out_pk_vly)(row.signal, row.ppg_pks, row.ppg_vlys, th=th_out) for i, row in df.iterrows())
	abp_out = df.out_abp.map(lambda s: len(s)!=0)
	ppg_out = df.out_ppg.map(lambda s: len(s)!=0)
	cl_log.write(" - removed by ABP outliers: {} \n".format(abp_out.sum()))
	cl_log.write(" - removed by PPG outliers: {} \n".format((ppg_out).sum()))

	df = df[~abp_out & ~ppg_out].reset_index(drop=True)

	return df

def _cycle_length_limitation(df, args, cl_log):
	lo_th_p2p = args.fs * args.cycle_len.lo_p2p
	up_th_p2p = args.fs * args.cycle_len.up_p2p

	removed_ppg_p2p = (df['ppg_p2p'] < lo_th_p2p) | (df['ppg_p2p'] > up_th_p2p) | (df['ppg_p2p'].isna())
	removed_abp_p2p = (df['abp_p2p'] < lo_th_p2p) | (df['abp_p2p'] > up_th_p2p) | (df['abp_p2p'].isna())
	removed_p2p = removed_ppg_p2p | removed_abp_p2p

	cl_log.write(" - removed by p2p distance: {} \n".format((removed_p2p).sum()))
	cl_log.write(" 	* removed by ABP p2p distance: {} \n".format(removed_abp_p2p.sum()))
	cl_log.write(" 	* removed by PPG p2p distance: {} \n".format((removed_ppg_p2p).sum()))
	
	removed_ppg_v2v = (df['ppg_v2v'] < lo_th_p2p) | (df['ppg_v2v'] > up_th_p2p) | (df['ppg_v2v'].isna())
	removed_abp_v2v = (df['abp_v2v'] < lo_th_p2p) | (df['abp_v2v'] > up_th_p2p) | (df['abp_v2v'].isna())
	removed_v2v = removed_ppg_v2v | removed_abp_v2v

	cl_log.write(" - removed by v2v distance: {} \n".format((removed_v2v).sum()))
	cl_log.write(" 	* removed by ABP v2v distance: {} \n".format(removed_abp_v2v.sum()))
	cl_log.write(" 	* removed by PPG v2v distance: {} \n".format((removed_ppg_v2v).sum()))

	df = df[~removed_p2p & ~removed_v2v].reset_index(drop=True)

	return df

def _distorted_signal_elimination(df, args, cl_log, isFirst=True):
	th_p2p_std = args.distorted_th.th_p2p_std
	th_amp_std = {'abp': args.distorted_th.th_amp_abp_std, 'ppg': args.distorted_th.th_amp_ppg_std}

	name_tab = {'pks': 'p2p','vlys': 'v2v'}
	targets = ['pks', 'vlys'] if isFirst else ['pks']
	mask_th_good = pd.Series([True]*df.shape[0])

	for target in targets:
		for s_type in ['abp','ppg']:
			for stat in ['std','amp_std','amp_dif_std']:
				stat_name = "{}_{}_{}".format(s_type, target, stat) if stat != 'std' else "{}_{}_{}".format(s_type, name_tab[target], stat)
				th = th_amp_std[s_type] if stat != 'std' else th_p2p_std

				cond = df[stat_name] <= th
				mask_th_good = ((mask_th_good) & (cond))
				cl_log.write(" - removed by {}: {} \n".format(stat_name,(~cond).sum()))
	df = df[mask_th_good].reset_index(drop=True)

	return df

def _compute_BP_labels(df, args, cl_log):
	SP_DPs = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(_compute_sp_dp)(sig, fs=args.fs) for sig in df.abp_signal)
	for i, label in enumerate(['SP','DP','f1','f2','abp_pks','abp_vlys']):
		df[label] = [val[i] for val in SP_DPs]
	cl_log.write(" - removed by BP not computed: {} \n".format((df.SP==-1).sum()))
	df = df[df.SP!=-1].reset_index(drop=True)

	return df

def _compute_features_df(df, args):
	""" Compute features and output dataframe with features, and dataframe with cleaned raw signals. """

	assert df.ppg_f2.sum() == 0
	## --------- Generate second set of features ---------
	heads_feats = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(extract_feat_cycle)(row.cs, row.pks_norm, fs=args.fs) for i,row in df.iterrows())

	list_2 = []
	for i,(h,f) in enumerate(heads_feats):
	    if len(h)==0 or len(f) == 0:
	        list_2.append(i)

	list_series = [pd.Series(f, index=h, dtype='float64') for h,f in heads_feats]
	res_second = pd.DataFrame(list_series)

	## --------- Generate first set of features ---------
	heads_feats_first = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(extract_feat_original)(fsig, fs=args.fs, filtered = args.ppg_filter.enable, remove_start_end=args.remove_start_end) for fsig in df['fsignal'])

	list_2_1 = []
	for i,(h,f) in enumerate(heads_feats_first):
	    if len(h)==0 or len(f) == 0:
	        list_2_1.append(i)
	        
	list_series = [pd.Series(f, index=h, dtype='float64') for h,f in heads_feats_first]
	res = pd.DataFrame(list_series)

	template = pd.read_csv(args.path.feats_template)['columns'].values

	cols = []
	for c in template:
	    if c in res.columns:
	        cols.append(c)

	res_first=res[cols].copy()

	res_feats = pd.concat([res_first,res_second], axis=1)
	data_feats = df[['patient','trial','SP','DP']].copy()
	data_feats = pd.concat([data_feats, res_feats],axis=1)

	data_feats = data_feats.drop(index=list_2+list_2_1).reset_index(drop=True)
	df = df.drop(index=list_2+list_2_1).reset_index(drop=True)

	keep_mask = ~data_feats.isna()['bd']

	data_feats = data_feats[keep_mask].reset_index(drop=True)
	df = df[keep_mask].reset_index(drop=True)

	return data_feats, df


def main(args):

	os.makedirs(os.path.dirname(args.path.log), exist_ok=True)
	cl_log = open(args.path.log, "w") ## Read logging file

	df = pd.read_pickle(args.path.data) ## Read dataset
	original_columns = df.columns ## Save original columns

	_print_n_samples(df, cl_log, sent = "Original data: ")


	#---------- Filter ABP values ----------#
	_print_step("Abnormal ABP values",cl_log)
	df = _compute_naive_BP(df)
	df = _abnormal_BP(df, args, cl_log, isABP=True)
	_print_n_samples(df, cl_log)


	#---------- Peaks and valleys computation ----------#
	_print_step("Peaks and valleys outliers",cl_log)
	df = _all_peaks_valleys(df, args, cl_log)
	df = _peaks_valleys_outliers(df, args, cl_log)
	_print_n_samples(df, cl_log)

	#---------- Limitation of cycle length ----------#
	_print_step("Limitation of cycle length (p2p & v2v distance threshold ({}, {}) seconds)"
		.format(args.cycle_len.lo_p2p, args.cycle_len.up_p2p),cl_log)
	df = _compute_quality_idx(df)
	df = _cycle_length_limitation(df, args, cl_log)
	_print_n_samples(df, cl_log)


	#---------- Elimination of distorted signals ----------#
	_print_step("Elimination of distorted signals",cl_log)
	df = _distorted_signal_elimination(df, args, cl_log, isFirst=True)
	_print_n_samples(df, cl_log)

	#---------- Baseline Wander (BW) removal ----------#
	_print_step("Baseline Wander (BW) removal",cl_log)
	df['signal'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(_rm_baseline_wander)(row.signal, row.ppg_vlys) for i, row in df.iterrows())
	df['abp_signal'] = Parallel(n_jobs=args.parallel.n_jobs, verbose=args.parallel.verbose)(delayed(_rm_baseline_wander)(row.abp_signal, row.abp_vlys) for i, row in df.iterrows())
	_print_n_samples(df, cl_log)


	#---------- Refinement ----------#
	_print_step("Refinement",cl_log)

	#---------- Filter ABP values ----------#
	_print_step("Abnormal ABP values",cl_log)
	df = _compute_naive_BP(df)
	df = _abnormal_BP(df, args, cl_log, isABP=True)
	_print_n_samples(df, cl_log)


	#---------- Filter BP values ----------#
	_print_step("Abnormal BP values",cl_log)
	df = _compute_BP_labels(df, args, cl_log) #Compute BP labels
	df = _abnormal_BP(df, args, cl_log, isABP=False) #Limit BP labels
	_print_n_samples(df, cl_log)


	#---------- Limitation of cycle length ----------#
	_print_step("Limitation of cycle length (p2p & v2v distance threshold ({}, {}) seconds)"
		.format(args.cycle_len.lo_p2p, args.cycle_len.up_p2p),cl_log)
	df = _extract_ppg_cycles(df, args, cl_log)
	df = _compute_quality_idx(df)
	df = _cycle_length_limitation(df, args, cl_log)
	_print_n_samples(df, cl_log)

	#---------- Elimination of distorted signals ----------#
	_print_step("Elimination of distorted signals",cl_log)
	df = _distorted_signal_elimination(df, args, cl_log, isFirst=False)
	_print_n_samples(df, cl_log)


	#---------- Feature generation ----------#
	_print_step("Segment removal by feature generation",cl_log)
	data_feats, data_raw = _compute_features_df(df, args)
	original_columns=original_columns.insert(2,'DP')
	original_columns=original_columns.insert(2,'SP')
	data_raw=data_raw[original_columns]
	_print_n_samples(data_feats, cl_log)


	#---------- SQI removal ----------#
	_print_step("Segment removal by skewness SQI",cl_log)
	keep_mask = data_feats['SQI_skew'] > args.lo_sqi
	data_feats = data_feats[keep_mask].reset_index(drop=True)
	data_raw = data_raw[keep_mask].reset_index(drop=True)
	_print_n_samples(data_feats, cl_log)


	#---------- Save data ----------#
	cl_log.close()
	data_feats.to_pickle(args.path.save_name_feats)
	data_raw.to_pickle(args.path.save_name)


if __name__=="__main__":

	#---------- Read config file ----------#
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_file", type=str, help="Path for the config file", required=True) 
	args_m = parser.parse_args()

	## Read config file
	if os.path.exists(args_m.config_file) == False:         
	    raise RuntimeError("config_file {} does not exist".format(args_m.config_file))
	args = OmegaConf.load(args_m.config_file)

	main(args)

	






