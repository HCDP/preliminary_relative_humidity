"""
QAQC workflow includes:
1. Crude range limiter
2. Single-pass outlier filter (move to double pass if it doesn't throw out too much)
3. Normal ratio gap-filling
If fewer than 8 values after outlier filters, code exits. Not enough data.
Empty or sparse data series is filled into date column.
RH mapping routine reverts to final back up based on Tmin.
Patch note: 0.1
Replace sklearn-contrib-py-earth with rpy2 implementation of R package "earth"
"""
import sys
import os
import pickle
import pytz
import pandas as pd
import numpy as np
from os.path import exists
from datetime import datetime,timedelta
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri

#CONSTANTS--------------------------------------------------------------------
#MASTER_DIR = '/mnt/lustre/koa/koastore/hawaii_climate_risk_group/kodamak8/relhum/'
MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/relhum/'
#MASTER_DIR = '/home/kodamak8/nrt_testing/relhum_test/relhum/'
DEP_DIR = MASTER_DIR + 'daily/dependencies/'
WORKING_DIR = MASTER_DIR + 'working_data/'
PICK_DIR = DEP_DIR + 'pickle/'
PRED_DIR = DEP_DIR + 'predictors/'
GAPFILL_DIR = DEP_DIR + 'gapfill/'
TA_DIR = WORKING_DIR + 'TA_record/'
DATA_DIR = MASTER_DIR + 'data_outputs/tables/station_data/daily/'
INPUT_DATA_DIR = DATA_DIR + 'raw/statewide/'
OUTPUT_DATA_DIR = DATA_DIR + 'partial_filled/statewide/'
OUTPUT_FLAG_DIR = DATA_DIR + 'qc_flags/'
MASTER_LINK = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
MASTER_DF = pd.read_csv(MASTER_LINK)
META_COLS = list(MASTER_DF.set_index('SKN').columns)
BREAK_POINT = 1500
GAPFILL_STNS = {339.5:[107.4,278,279,339.4],339.6:[278,279,339.5]}
PRED_LIST = ['dem_250']
#END CONSTANTS----------------------------------------------------------------
#FUNCTIONS--------------------------------------------------------------------
earth = rpackages.importr("earth") #Keep to global scope for now, hopefully this doesn't bug out
def load_pickle(pick_name):
    with open(pick_name,'rb') as f:
        pick_models = pickle.load(f)
    return pick_models

def vapor_pressure(td):
    c1 = 6.1078
    c2 = 17.269
    c3 = 237.3
    frac = td/(td+c3)
    return c1*np.exp(c2*frac)

def compute_rh(vp,es):
    rh_pre = 100*vp/es
    inds = np.where(rh_pre>100)
    rh_pre[inds] = 100
    return rh_pre

def earth_to_mars(x_fit,y_fit,newdata,pred_list=['elev']):
    #x_fit: numpy array with training independent data
    #y_fit: numpy array with training dependent data
    #newdata: numpy array of independent data to generate new prediction
    #pred_list: string list of predictor names, defaults to 'elev' but required for all other vars
    #Creating input pandas dataframe
    indep_df = pd.DataFrame([x_fit[i] for i in range(x_fit.shape[0])],columns=pred_list)
    dep_df = pd.DataFrame(y_fit,columns=['TD'])
    training_py = pd.concat([indep_df,dep_df],axis=1)
    nd_df = pd.DataFrame([newdata[i] for i in range(newdata.shape[0])],columns=pred_list)
    #Convert to r.data.frame
    with (ro.default_converter+pandas2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(training_py)
        r_nd = ro.conversion.get_conversion().py2rpy(nd_df)
    #Create R earth object with training data
    model = earth.earth(formula=Formula("TD~."),data=r_df)
    #Generate prediction using newdata
    prediction = np.squeeze(np.asarray(ro.r.predict(model,newdata=r_nd)))
    return prediction

def split_dataframe(df,meta_cols,dt_convert=True):
    df = df.set_index('SKN')
    data_cols = [col for col in list(df.columns) if col not in meta_cols]
    df_meta = df[meta_cols]
    df_data = df[data_cols]
    if dt_convert:
        date_cols = pd.to_datetime([col.split('X')[1] for col in data_cols])
    else:
        date_cols = pd.to_datetime(data_cols)
    df_data.columns = date_cols
    return df_data,df_meta

def sort_dates(df,meta_cols):
    non_meta_cols = [col for col in list(df.columns) if col not in meta_cols]
    date_keys_sorted = sorted(pd.to_datetime([dt.split('X')[1] for dt in non_meta_cols]))
    date_cols_sorted = [dt.strftime('X%Y.%m.%d') for dt in date_keys_sorted]
    sorted_cols = meta_cols + date_cols_sorted
    sorted_df = df[sorted_cols]
    return sorted_df

def update_csv(new_data,csv_name,date_str=None):
    #Add metadata
    #cross check new data with master meta, exclude invalid stations
    new_checked = np.intersect1d(new_data.index.values,MASTER_DF.set_index('SKN').index.values)
    new_data = new_data.loc[new_checked]
    new_meta = MASTER_DF.set_index('SKN').loc[new_data.index]
    new_merged = new_meta.join(new_data,how='left')

    if exists(csv_name):
        if (os.stat(csv_name).st_size != 0):
            old_df = pd.read_csv(csv_name)
            old_df = old_df.set_index('SKN')
            old_inds = old_df.index.values
            old_cols = list(old_df.columns)
            #Cross check old SKNs to make sure prior invalids don't repopulate
            old_checked = np.intersect1d(old_inds,MASTER_DF.set_index('SKN').index.values)
            old_df = old_df.loc[old_checked]
            upd_inds = np.union1d(old_checked,new_merged.index.values)
            #Patch 07/2024: include all dates even if empty
            if date_str != None:
                current_day =pd.to_datetime(date_str)
                mon_st = current_day.to_period('M').to_timestamp()
                days_to_date = pd.date_range(mon_st,current_day)
                date_col_fmt = [dt.strftime('X%Y.%m.%d') for dt in days_to_date]
                upd_df = pd.DataFrame(index=upd_inds,columns=date_col_fmt)
            else:
                upd_df = pd.DataFrame(index=upd_inds)
            upd_df.index.name = 'SKN'
            #Backfill
            upd_df.loc[old_checked,old_cols] = old_df
            #Fill new
            upd_df.loc[new_merged.index,new_merged.columns] = new_merged
            #Write new csv
            upd_df = sort_dates(upd_df,META_COLS)
            upd_df = upd_df.fillna('NA')
            upd_df.to_csv(csv_name)
        else:
            #If it's empty, just overwrite with new dataframe
            new_merged = new_merged.fillna('NA')
            new_merged.to_csv(csv_name)
    else:
        #Write new csv if file doesn't exist OR file exists but is empty
        new_merged = new_merged.fillna('NA')
        new_merged.to_csv(csv_name)

def myModel(inversion=2150):
    '''
    This wrapper function constructs another function called "MODEL"
    according to the provided inversion elevation
    '''

    def MODEL(X, *theta):

        _, n_params = X.shape

        y = theta[0] + theta[1] * X[:, 0]
        for t in range(1, n_params):
            y += theta[t+2] * X[:, t]

        ind, = np.where(X[:, 0] > inversion)
        y[ind] += theta[2] * (X[:, 0][ind] - inversion)

        return y

    return MODEL

def range_limit(z,td):
    """
    Based on a pre-determined range (based on historical TD data), check if
    data point is within range (at respective elevation).
    *Note: if further predictors added, this becomes more complicated.*
    Input:
    -Z: elev series with SKN index
    -TD: corresponding TD series with SKN index
    Output array of flags (0 out of range, 1 in range) to filter out-of-
    -range values.
    """
    pick_file = PICK_DIR + 'td_range_pickle.dat'
    MODEL = myModel(inversion=BREAK_POINT)
    range_model = load_pickle(pick_file)
    std = range_model['std'][0]
    theta_top = range_model['thetas'][0]
    theta_bot = range_model['thetas'][1]
    #z values should correspond to an interval
    range_top = MODEL(z.values.reshape(-1,1),*theta_top)
    range_bot = MODEL(z.values.reshape(-1,1),*theta_bot)
    td_flags = pd.Series(np.zeros(range_top.shape),index=td.index)
    td_flags[(td>=range_bot)&(td<=range_top)] = 1
    return td_flags

def single_pass_outlier(pred,vals,threshold=2.5):
    #Does an initial fit and then removes those outside the threshold range
    val_est = earth_to_mars(pred,vals,pred,pred_list=['elev'])
    stdev = threshold*np.std(val_est-vals)
    ind_within = np.where(np.abs(val_est-vals)<stdev)
    pred_pass = pred[ind_within]
    vals_pass = vals[ind_within]
    pred_pass = pred_pass.squeeze()
    return ind_within,pred_pass,vals_pass

def normal_ratio(td_day):
    td_filled = td_day.copy()
    mean_table = GAPFILL_DIR + 'TD_20yr_mean_nr.csv'
    means_df = pd.read_csv(mean_table)
    means_df = means_df.set_index('SKN')['Mean']
    avail_stns = td_day.index.values
    ess_stns = list(GAPFILL_STNS.keys())
    missing_stns = np.setdiff1d(ess_stns,avail_stns)
    if len(missing_stns) > 0:
        for targ_stn in missing_stns:
            Ntarg = means_df.loc[targ_stn]
            donor_stns = GAPFILL_STNS[targ_stn]
            avail_donors = np.intersect1d(td_day.index.values,donor_stns)
            if len(np.intersect1d(avail_donors,donor_stns)) > 0:
                Ndonor = means_df.loc[avail_donors]
                Nratio = Ntarg/Ndonor
                donor_df = td_day.loc[avail_donors]
                ratiod = donor_df * Nratio
                targ_est = ratiod.mean()
                td_filled.loc[targ_stn] = targ_est
            else:
                #No donors, fill with mean value
                td_filled.loc[targ_stn] = means_df[targ_stn]
                
    #If no missing stations, returns original
    #If missing, run the gapfilling procedure first and a process has occurred
    return td_filled

def raw_rh_file(date_str):
    #Uses the TD and TA to create pre-qc RH file
    dt = pd.to_datetime(date_str)
    date_fmt = dt.strftime('X%Y.%m.%d')
    year_str = dt.strftime('%Y')
    mon_str = dt.strftime('%m')
    td_file = INPUT_DATA_DIR + '_'.join(('daily','TD',year_str,mon_str))+'.csv'
    td_df = pd.read_csv(td_file)
    td_data,td_meta = split_dataframe(td_df,META_COLS)
    ta_file = TA_DIR + '_'.join(('daily','TA',year_str,mon_str))+'.csv'
    ta_df = pd.read_csv(ta_file)
    ta_data,ta_meta = split_dataframe(ta_df,META_COLS)
    if (dt in list(td_data.columns))&(dt in list(ta_data.columns)):
        td_day = td_data[dt].dropna()
        ta_day = ta_data[dt].dropna()
        if (td_day.shape[0]>0)&(ta_day.shape[0]>0):            
            #match them
            match_skn = np.intersect1d(td_day.index.values,ta_day.index.values)
            td_match = td_day.loc[match_skn]
            ta_match = ta_day.loc[match_skn]
            vp = vapor_pressure(td_match)
            es = vapor_pressure(ta_match)
            rh_arr = compute_rh(vp.values,es.values)
            rh = pd.Series(rh_arr,index=vp.index)
            rh.name = date_fmt
        else:
            rh = pd.Series(name=date_fmt)
    else:
        rh = pd.Series(name=date_fmt)
    rh_file = INPUT_DATA_DIR + '_'.join(('daily','RH',year_str,mon_str))+'.csv'
    update_csv(rh,rh_file,date_str=date_str)


def qaqc(date_str):
    dt = pd.to_datetime(date_str)
    date_fmt = dt.strftime('X%Y.%m.%d')
    year_str = dt.strftime('%Y')
    mon_str = dt.strftime('%m')
    pred_file = PRED_DIR + 'updated_TD_predictors.csv'
    pred_df = pd.read_csv(pred_file)
    pred_df = pred_df.set_index('SKN')
    td_file = INPUT_DATA_DIR + '_'.join(('daily','TD',year_str,mon_str))+'.csv'
    td_df = pd.read_csv(td_file)
    td_data,td_meta = split_dataframe(td_df,META_COLS)
    if dt in list(td_data.columns):
        td_day = td_data[dt].dropna()
        td_day.name = date_fmt
        elev_day = td_meta.loc[td_day.index,'ELEV.m.']
    else:
        #Fill day with empty column
        td_day = pd.Series(name=date_fmt)
        td_final_flags = pd.Series(name=date_fmt)
    if td_day.shape[0] > 0:
        #Range limiter
        orig_inds = td_day.index.values
        td_flags = range_limit(elev_day,td_day)
        td_filt1 = td_day.copy()
        td_filt1 = td_filt1[td_flags>0]
        #Linear fit outlier pass
        ##match preds by intersect
        match_preds_ind = np.intersect1d(pred_df.index.values,td_filt1.index.values)
        match_preds = pred_df.loc[match_preds_ind,PRED_LIST]
        td_filt1 = td_filt1.loc[match_preds_ind]
        ifilt,pred_filt,out_filt = single_pass_outlier(match_preds.values,td_filt1.values)
        td_filt2 = td_filt1.copy().iloc[ifilt]
        #Gapfill
        td_day = normal_ratio(td_filt2)
        #Set QC flags
        final_inds = np.union1d(orig_inds,td_day.index.values)
        td_final_flags = pd.DataFrame(np.zeros(final_inds.shape),index=final_inds,columns=[date_fmt])
        #Range limited - flag == 1
        filt1_inds = td_flags[td_flags==0].index.values
        td_final_flags.loc[filt1_inds] = 1
        #Regression filtered - flag == 2
        filt2_inds = np.setdiff1d(td_filt1.index.values,td_filt2.index.values)
        td_final_flags.loc[filt2_inds] = 2
        #Gapfilled values - flag == -1
        filled_inds = np.setdiff1d(td_day.index.values,td_filt2.index.values)
        td_final_flags.loc[filled_inds] = -1
    else:
        td_day = pd.Series(name=date_fmt)
        td_final_flags = pd.Series(name=date_fmt)
    td_day.index.name = 'SKN'
    td_final_flags.index.name = 'SKN'
    #Create RH file based on TA
    ta_file = TA_DIR + '_'.join(('daily','TA',year_str,mon_str))+'.csv'
    ta_df = pd.read_csv(ta_file)
    ta_data,ta_meta = split_dataframe(ta_df,META_COLS)
    if dt in list(ta_data.columns):
        ta_day = ta_data[dt].dropna()
        if ta_day.shape[0]>0:
            match_skns = np.intersect1d(ta_day.index.values,td_day.index.values)
            ta_match = ta_day.loc[match_skns]
            td_match = td_day.loc[match_skns]
            vp = vapor_pressure(td_match)
            es = vapor_pressure(ta_match)
            rh_arr = compute_rh(vp.values,es.values)
            rh = pd.Series(rh_arr,index=vp.index)
            rh.name = date_fmt
        else:
            rh = pd.Series(name=date_fmt)
    else:
        rh = pd.Series(name=date_fmt)
    rh_file = OUTPUT_DATA_DIR + '_'.join(('daily','RH',year_str,mon_str,'qc'))+'.csv'
    update_csv(rh,rh_file,date_str=date_str)
    #td_day is either empty series or it's filtered and partially filled
    #update the qc csv with qced td_day
    #also create a qc flag file with the same indices as td_filled
    output_file = OUTPUT_DATA_DIR + '_'.join(('daily','TD',year_str,mon_str,'qc'))+'.csv'
    flag_file = OUTPUT_FLAG_DIR + '_'.join(('daily','TD',year_str,mon_str,'flag'))+'.csv'
    update_csv(td_day,output_file,date_str=date_str)
    update_csv(td_final_flags,flag_file,date_str=date_str)

        
    
#END FUNCTIONS----------------------------------------------------------------
if __name__=="__main__":
    if len(sys.argv) > 1:
        input_date = sys.argv[1]
        dt = pd.to_datetime(input_date)
        date_str = dt.strftime('%Y-%m-%d')
    else:
        #Set date to previous 24 hours
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        prev_day = today - timedelta(days=1)
        date_str = prev_day.strftime('%Y-%m-%d')
    print(date_str)
    raw_rh_file(date_str)
    qaqc(date_str)