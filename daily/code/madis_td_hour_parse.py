import sys
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from os.path import exists

#DEFINE CONSTANTS--------------------------------------------------------------
SOURCE_NAME = 'madis'
SRC_KEY = 'stationId'
SRC_TIME = 'time'
MASTER_KEY = 'NWS.id'
#MASTER_DIR = r'/home/kodamak8/nrt_testing/relhum_test/'
MASTER_DIR = r'/home/hawaii_climate_products_container/preliminary/'
PARSE_DIR = r'/home/hawaii_climate_products_container/preliminary/data_aqs/data_outputs/' + SOURCE_NAME + r'/parse/'
PROCESS_DIR = MASTER_DIR + r'relhum/working_data/processed_data/'
TRACK_DIR = MASTER_DIR + r'relhum/data_outputs/tables/relhum_station_tracking/'
HOUR_DEFAULT = 13
INT_EXCEPT = {'AR427':8}
MASTER_LINK = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
#END CONSTANTS-----------------------------------------------------------------

#DEFINE FUNCTIONS--------------------------------------------------------------
def inverse_rh(rh,ta):
    rh = rh/100
    es = vapor_pressure(ta)
    vp = rh*es
    a = 6.1078
    b = 17.269
    c = 237.3
    frac = vp/a
    td = c*np.log(frac)/(b - np.log(frac))
    return td

def vapor_pressure(td):
    c1 = 6.1078
    c2 = 17.269
    c3 = 237.3
    frac = td/(td+c3)
    return c1*np.exp(c2*frac)

def get_max_counts(temp_df,uni_stns):
    max_counts = {}
    for stn in uni_stns:
        if stn in INT_EXCEPT.keys():
            max_count = INT_EXCEPT[stn]
        else:
            stn_df = temp_df[temp_df[SRC_KEY]==stn].sort_values(by=SRC_TIME)
            stn_times = pd.to_datetime(stn_df[SRC_TIME])
            stn_ints = (stn_times.round('min').diff().dropna().dt.seconds/60).values
            if len(stn_ints) < 1:
                continue
            vals, counts = np.unique(stn_ints,return_counts=True)
            mode_id = np.argmax(counts)
            mode_val = vals[mode_id]
            max_count = 60/mode_val
        max_counts[stn] = max_count
    return max_counts

def get_hour_td(df,date_str,hour=HOUR_DEFAULT):
    #Get td measurements
    td_df = df[df['varname'].isin(['dewpoint'])]
    td_df = td_df[['stationId','time','varname','value']]
    td_df = td_df[td_df['value'].notnull()]
    #Convert rh to td
    rh_df = df[df['varname'].isin(['relHumidity'])]
    ta_df = df[df['varname'].isin(['temperature'])]
    static_cols = ['stationId','stationName','dataProvider','time','source']
    rh_ta_df = rh_df.merge(ta_df,on=static_cols,how='inner').rename(columns={'value_x':'relHumidity','value_y':'temperature'})
    converted_td = inverse_rh(rh_ta_df['relHumidity'],rh_ta_df['temperature'])
    converted_td.name = 'value'
    converted_td_meta = rh_ta_df[['stationId','time']]
    var_col = pd.Series(['dewpoint'])
    var_col = var_col.repeat(converted_td.shape[0]).reset_index(drop=True)
    var_col.name = 'varname'
    converted_td = pd.concat([converted_td_meta,var_col,converted_td],axis=1)
    converted_td = converted_td[converted_td['value'].notnull()]
    td_all = pd.concat([td_df,converted_td],axis=0)
    #Remove duplicates
    td_undup = td_all.drop_duplicates(subset=['stationId','time']).reset_index(drop=True)
    uni_stns = td_undup[SRC_KEY].unique()
    td_hour_data = []
    hour_hst = pd.to_datetime(date_str)+ pd.Timedelta(hours=13)
    hour_utc = hour_hst + pd.Timedelta(hours=10)
    hour_st = hour_utc
    hour_en = hour_utc + pd.Timedelta(hours=1)
    max_cts = get_max_counts(td_undup,uni_stns)
    date_str = pd.to_datetime(date_str).strftime('X%Y.%m.%d')
    for stn in uni_stns:
        if stn in list(max_cts.keys()):
            stn_df = td_undup[td_undup[SRC_KEY]==stn]
            obs_time = pd.to_datetime(stn_df[SRC_TIME])
            hour_select = obs_time[(obs_time>hour_st)&(obs_time<=hour_en)]
            stn_max_count = max_cts[stn]
            valid_pct = hour_select.shape[0]/stn_max_count
            td_select = td_undup.loc[hour_select.index,'value'].mean()
            td_hour_data.append([stn,'TD',date_str,td_select,valid_pct])
    td_hour_df = pd.DataFrame(td_hour_data,columns=[MASTER_KEY,'var','date','value','percent_valid'])
    #Get TA in similar fashion
    ta_uni = ta_df[SRC_KEY].unique()
    ta_max_cts = get_max_counts(ta_df,uni_stns)
    ta_hour_data = []
    for stn in ta_uni:
        if stn in list(ta_max_cts.keys()):
            stn_df = ta_df[ta_df[SRC_KEY]==stn]
            obs_time = pd.to_datetime(stn_df[SRC_TIME])
            hour_select = obs_time[(obs_time>hour_st)&(obs_time<=hour_en)]
            stn_max_count = ta_max_cts[stn]
            valid_pct = hour_select.shape[0]/stn_max_count
            ta_select = ta_df.loc[hour_select.index,'value'].mean()
            ta_hour_data.append([stn,'TA',date_str,ta_select,valid_pct])
    ta_hour_df = pd.DataFrame(ta_hour_data,columns=[MASTER_KEY,'var','date','value','percent_valid'])
    return td_hour_df,ta_hour_df

def convert_dataframe(long_df,varname):
    var_df = long_df[long_df['var']==varname]
    valid_df = var_df[var_df['percent_valid']>=0.95]
    wide_df = pd.DataFrame(index=valid_df[MASTER_KEY].values)
    for stn in wide_df.index.values:
        stn_temp = valid_df[valid_df[MASTER_KEY]==stn].set_index('date')[['value']]
        wide_df.loc[stn,stn_temp.index.values] = stn_temp['value']
    
    wide_df.index.name = MASTER_KEY
    wide_df = wide_df.reset_index()
    return wide_df

def update_csv(csv_name,new_data_df):
    master_df = pd.read_csv(MASTER_LINK)
    prev_ids = new_data_df[MASTER_KEY].values
    merged_new_df = master_df.merge(new_data_df,on=MASTER_KEY,how='inner')
    merged_new_df = merged_new_df.set_index('SKN')
    merged_ids = merged_new_df[MASTER_KEY].values
    unkn_ids = np.setdiff1d(prev_ids,merged_ids)
    master_df = master_df.set_index('SKN')
    meta_cols = list(master_df.columns)
    if exists(csv_name):
        old_df = pd.read_csv(csv_name)
        old_df = old_df.set_index('SKN')
        old_cols = list(old_df.columns)
        old_inds = old_df.index.values
        upd_inds = np.union1d(old_inds,merged_new_df.index.values)
        updated_df = pd.DataFrame(index=upd_inds)
        updated_df.index.name = 'SKN'
        updated_df.loc[old_inds,old_cols] = old_df
        updated_df.loc[merged_new_df.index.values,merged_new_df.columns] = merged_new_df
        updated_df = sort_dates(updated_df,meta_cols)
        updated_df = updated_df.fillna('NA')
        updated_df = updated_df.reset_index()
        updated_df.to_csv(csv_name,index=False)
    else:
        merged_new_df = merged_new_df.fillna('NA')
        merged_new_df = merged_new_df.reset_index()
        merged_new_df.to_csv(csv_name,index=False)
    
    return unkn_ids

def sort_dates(df,meta_cols):
    non_meta_cols = [col for col in list(df.columns) if col not in meta_cols]
    date_keys_sorted = sorted(pd.to_datetime([dt.split('X')[1] for dt in non_meta_cols]))
    date_cols_sorted = [dt.strftime('X%Y.%m.%d') for dt in date_keys_sorted]
    sorted_cols = meta_cols + date_cols_sorted
    sorted_df = df[sorted_cols]
    return sorted_df

def update_unknown(unknown_file,unknown_ids,date_str):
    if exists(unknown_file):
        prev_df = pd.read_csv(unknown_file)
        preex_ids = np.intersect1d(unknown_ids,list(prev_df['sourceID'].values))
        new_ids = np.setdiff1d(unknown_ids,list(prev_df['sourceID'].values))
        prev_df = prev_df.set_index('sourceID')
        prev_df.loc[preex_ids,'lastDate'] = date_str
        prev_df = prev_df.reset_index()
        data_table = [[new_ids[i],SOURCE_NAME,date_str] for i in range(len(new_ids))]
        unknown_df = pd.DataFrame(data_table,columns=['sourceID','datastream','lastDate'])
        prev_df = pd.concat([prev_df,unknown_df],axis=0,ignore_index=True)
        prev_df.to_csv(unknown_file,index=False)
    else:
        data_table = [[unknown_ids[i],SOURCE_NAME,date_str] for i in range(len(unknown_ids))]
        unknown_df = pd.DataFrame(data_table,columns=['sourceID','datastream','lastDate'])
        unknown_df.to_csv(unknown_file,index=False)

def get_station_sorted_data(date_str,hour=HOUR_DEFAULT):
    date_code = pd.to_datetime(date_str).strftime('%Y%m%d')
    date_year = pd.to_datetime(date_str).strftime('%Y')
    date_mon = pd.to_datetime(date_str).strftime('%m')
    source_file = PARSE_DIR + '_'.join((date_code,SOURCE_NAME,'parsed'))+'.csv'
    df = pd.read_csv(source_file)
    td_hour_df,ta_hour_df = get_hour_td(df,date_str,hour=hour)
    td_wide = convert_dataframe(td_hour_df,'TD')
    ta_wide = convert_dataframe(ta_hour_df,'TA')
    td_proc_file = PROCESS_DIR + SOURCE_NAME + '/' + '_'.join(('TD',SOURCE_NAME,date_year,date_mon,'processed'))+'.csv'
    ta_proc_file = PROCESS_DIR + SOURCE_NAME + '/' + '_'.join(('TA',SOURCE_NAME,date_year,date_mon,'processed'))+'.csv'
    unknown_ids = update_csv(td_proc_file,td_wide)
    scratch_ids = update_csv(ta_proc_file,ta_wide)

    unknown_file = TRACK_DIR + '_'.join(('unknown_TD_sta',date_year,date_mon))+'.csv'
    update_unknown(unknown_file,unknown_ids,date_str)


#END FUNCTIONS-----------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) == 1:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        prev_day = today - timedelta(days=1)
        date_str = prev_day.strftime('%Y-%m-%d')
        hour = HOUR_DEFAULT
    elif (len(sys.argv) > 1)&(len(sys.argv)<=2):
        date_str = sys.argv[1]
        hour = HOUR_DEFAULT
    else:
        date_str = sys.argv[1]
        hour = sys.argv[2]
    get_station_sorted_data(date_str,hour=hour)