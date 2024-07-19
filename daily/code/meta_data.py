import sys
import pickle
import pytz
import rasterio
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import r2_score
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri

#Batch directories
#MASTER_DIR = '/mnt/lustre/koa/koastore/hawaii_climate_risk_group/kodamak8/'
#Realtime directories
MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/'
#MASTER_DIR = '/home/kodamak8/nrt_testing/relhum_test/'
DEP_DIR = MASTER_DIR + 'relhum/daily/dependencies/'
PICK_DIR = DEP_DIR + 'pickle/'
PRED_DIR = DEP_DIR + 'predictors/'
OUTPUT_MASTER_DIR = MASTER_DIR + 'relhum/data_outputs/'
TD_DIR = OUTPUT_MASTER_DIR + 'tables/station_data/daily/partial_filled/statewide/'
TA_DIR = MASTER_DIR + 'relhum/working_data/TA_record/'
#TMAX_DIR = DEP_DIR + 'gridded_temp/'
TMAX_DIR = DEP_DIR + 'gridded_temp/'
QC_DIR = OUTPUT_MASTER_DIR + 'tables/station_data/daily/qc_flags/'
CV_DIR = OUTPUT_MASTER_DIR + 'tables/loocv/daily/'
META_DIR = OUTPUT_MASTER_DIR + 'metadata/daily/'
MASTER_LINK = r'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
MASTER_DF = pd.read_csv(MASTER_LINK)
META_COLS = list(MASTER_DF.set_index('SKN').columns)
ISL_LIST = {'BI':['BI'],'KA':['KA'],'MN':['MA','MO','KO','LA'],'OA':['OA'],'state':['BI','KA','MA','MO','KO','LA','OA']}
PRED_LIST = ['dem_250']
FORMAL_NAME = {'BI':r"Hawai'i County",'KA':r"Kaua'i County",'MN':r"Maui County",'OA':r"Honolulu County",'state':r"the State of Hawai'i"}
VERSION = 'preliminary'

earth = rpackages.importr("earth") #Keep to global scope for now, hopefully this doesn't bug out
#Design this to run in batch and real-time based on inputs
def linear(x,a,b):
    return a*x+b

def load_pickle(pick_name):
    with open(pick_name,'rb') as f:
        pick_data = pickle.load(f)
    return pick_data

def vapor_pressure(td):
    c1 = 6.1078
    c2 = 17.269
    c3 = 237.3
    frac = td/(td+c3)
    return c1*np.exp(c2*frac)

def compute_rh(vp,es):
    rh_pre = 100*vp/es
    if rh_pre.ndim > 0:
        inds = np.where(rh_pre>100)
        rh_pre[inds] = 100
    else:
        if rh_pre > 100:
            rh_pre = 100

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

def cross_validation(dt,icode):
    yearmon_str = dt.strftime('%Y_%m')
    tmax_date = dt.strftime('%Y%m%d')
    #Get TD data
    td_file = TD_DIR + '_'.join(('daily','TD',yearmon_str,'qc'))+'.csv'
    td_df = pd.read_csv(td_file)
    td_data,td_meta = split_dataframe(td_df,META_COLS)
    td_day = td_data[dt].dropna()
    td_day_meta = td_meta.loc[td_day.index]
    val_stns = td_day_meta[td_day_meta['Island'].isin(ISL_LIST[icode])].index.values
    #Get corresponding TA station data
    ta_file = TA_DIR + '_'.join(('daily','TA',yearmon_str))+'.csv'
    ta_df = pd.read_csv(ta_file)
    ta_data,ta_meta = split_dataframe(ta_df,META_COLS)
    ta_day = ta_data[dt].dropna()
    match_stns = np.intersect1d(val_stns,ta_day.index.values)
    #Extract appropriate Tmax data from raster for RH estimate
    tmax_pick_file = PICK_DIR + 'ta_model_pickle.dat'
    tmax_pickle = load_pickle(tmax_pick_file)
    tmax_theta = tmax_pickle['thetas'][0]
    #tmax_rast_file = TMAX_DIR + '_'.join(('Tmax','map',icode,tmax_date))+'.tif'
    tmax_rast_file = TMAX_DIR + '_'.join(('Tmax','map',icode,tmax_date))+'.tif'
    tmax_raster = rasterio.open(tmax_rast_file)
    tmax_data = tmax_raster.read(1)
    #Get predictor
    pred_file = PRED_DIR + 'updated_TD_predictors.csv'
    pred_df = pd.read_csv(pred_file)
    pred_df = pred_df.set_index('SKN')
    cv_stats = []
    for stn in match_stns:
        #Get estimated TA at validation stn
        stn_lon = ta_meta.at[stn,'LON']
        stn_lat = ta_meta.at[stn,'LAT']
        py,px = tmax_raster.index(stn_lon,stn_lat)
        stn_tmax = tmax_data[py,px]
        est_ta = linear(stn_tmax,*tmax_theta)
        #Estimate TD
        val_ta = ta_day.loc[stn]
        val_data = td_day.loc[stn]
        val_pred = pred_df.loc[stn,PRED_LIST]
        training_inds = np.setdiff1d(td_day.index.values,[stn])
        train_data = td_day.loc[training_inds]
        train_pred = pred_df.loc[training_inds,PRED_LIST]
        est_data = earth_to_mars(train_pred.values,train_data.values,val_pred.values)
        #Convert validation and estimate to RH using actual and estimated TA
        val_vp = vapor_pressure(val_data)
        val_es = vapor_pressure(val_ta)
        val_rh = compute_rh(val_vp,val_es)
        est_vp = vapor_pressure(est_data)
        est_es = vapor_pressure(est_ta)
        est_rh = compute_rh(est_vp,est_es)
        td_anom = val_data - est_data
        rh_anom = val_rh - est_rh
        cv_stats.append([stn,val_data,est_data,val_rh,est_rh,td_anom,rh_anom])
    #Convert array into df
    cv_df = pd.DataFrame(cv_stats,columns=['SKN','TD_Obs','TD_Pred','RH_Obs','RH_Pred','TD_Obs-Pred','RH_Obs-Pred'])
    cv_df = cv_df.set_index('SKN')
    return cv_df

def get_metrics(cv_stats):
    #Fetches RH metrics specifically
    #metrics needed: RMSE, bias, R2, MAE
    rmse = np.sqrt(np.mean((cv_stats['RH_Obs'].values-cv_stats['RH_Pred'].values)**2))
    bias = np.mean((cv_stats['RH_Obs-Pred'].values))
    mae = np.mean(np.abs(cv_stats['RH_Obs-Pred'].values))
    r2 = r2_score(cv_stats['RH_Obs'].values,cv_stats['RH_Pred'].values)
    metrics_dict = {'rmse':rmse,'bias':bias,'mae':mae,'r2':r2}
    return metrics_dict

def format_metadata(metrics,dt,icode):
    #Format dates
    date_fmt = dt.strftime('%b. %d, %Y')
    date_str = dt.strftime('%Y%m%d')
    yearmon_str = dt.strftime('%Y_%m')
    if icode == 'state':
        locality = 'statewide/'
    else:
        locality = 'county/'+icode.upper()+'/'
    #Get static values
    static_pickle = PICK_DIR + 'fixed_metrics_by_isl.dat'
    static_meta = load_pickle(static_pickle)
    fixed_res = str(static_meta['res'])
    fixed_proj = static_meta['proj']
    fixed_coord = static_meta['coord']
    fixed_fill = static_meta['fill']
    xmin,ymin = static_meta['minmax'][icode][0]
    xmax,ymax = static_meta['minmax'][icode][1]
    if icode == 'state':
        npix = ', '.join([str(pix) for pix in list(static_meta['pix'].values())])
    else:
        npix = str(static_meta['pix'][icode])

    #Format file names
    data_file = '_'.join(('daily','RH',yearmon_str,'qc'))+'.csv'
    meta_file = '_'.join((date_str,'RH',icode,'meta'))+'.txt'
    cv_file = '_'.join((date_str,'RH',icode,'loocv'))+'.csv'
    grid_file = '_'.join(('RH','map',icode,date_str))+'.tif'
    #Format metrics
    rmse_fmt = str(np.round(metrics['rmse'],3))
    bias_fmt = str(np.round(metrics['bias'],3))
    mae_fmt = str(np.round(metrics['mae'],3))
    r2_fmt = str(np.round(metrics['r2'],3))

    #Get station metrics
    qc_file = QC_DIR + '_'.join(('daily','TD',yearmon_str,'flag'))+'.csv'
    td_file = TD_DIR + '_'.join(('daily','TD',yearmon_str,'qc'))+'.csv'
    qc_df = pd.read_csv(qc_file)
    td_df = pd.read_csv(td_file)
    qc_data,qc_meta = split_dataframe(qc_df,META_COLS)
    td_data,td_meta = split_dataframe(td_df,META_COLS)
    qc_day = qc_data[dt]
    td_day = td_data[dt].dropna()
    nstns = td_day.shape[0]
    ngap = qc_day[qc_day==-1].shape[0]
    
    #Data statement
    intro_text = "This daily relative humidity map of {ext} is a high spatial resolution gridded prediction of relative humidity in {units} for the date {date}."
    intro_fmt = intro_text.format(ext=FORMAL_NAME[icode],units="percent humidity",date=date_fmt)
    method_text1 = "This was produced using a Multivariate Adaptive Regression Spline model, which was trained on {nstn} unique stations statewide and used to estimate dewpoint temperature, predicted by elevation."
    method_fmt1 = method_text1.format(nstn=str(nstns))
    #Gapfill statement
    if ngap > 0:
        gapfill_text = "Missing data were gapfilled for {nfilled} high-elevation stations using the Normal Ratio method."
        gapfill_fmt = gapfill_text.format(nfilled=str(ngap))
    else:
        gapfill_fmt = ""
    method_text2 = "Dewpoint temperature estimates are converted to percent relative humidity by using estimates of concurrent 2 meter air temperature to calculate saturation vapor pressure. The air temperature is estimated using daily maximum temperature for {date}, available on the Hawai'i Climate Data Portal."
    method_fmt2 = method_text2.format(date=date_fmt)
    rsq_text = "A leave-one-out cross-validation (LOOCV) of the station data used in this map produced an R-squared of: {rsq}."
    rsq_fmt = rsq_text.format(rsq=r2_fmt)
    err_text = "All maps are subject to change as new data becomes available or unknown errors are corrected in reoccurring versions. Errors in humidity estimates do vary over space meaning any gridded humidity value, even on higher quality maps, could still produce incorrect estimates."
    data_statement = ' '.join((intro_fmt,method_fmt1,gapfill_fmt,method_fmt2,rsq_fmt,err_text))

    kw_list = ', '.join([FORMAL_NAME[icode],'Hawaii',' humidity prediction','daily relative humidity','relative humidity','climate','linear regression'])
    credit_statement = 'All data produced by University of Hawaii at Manoa Dept. of Geography and the Enviroment, Ecohydology Lab in collaboration with the Water Resource Research Center (WRRC). Support for the Hawai‘i EPSCoR Program is provided by the Hawaii Emergency Management Agency.'
    contact_list = 'Keri Kodama (kodamak8@hawaii.edu), Matthew Lucas (mplucas@hawaii.edu), Ryan Longman (rlongman@hawaii.edu), Sayed Bateni (smbateni@hawaii.edu), Thomas Giambelluca (thomas@hawaii.edu)'

    field_val_list = {'attribute':'value','dataStatement':data_statement,'keywords':kw_list,
                      'domain':icode.lower(),'date':date_fmt,'versionType':VERSION,
                      'stationFile':data_file,'gridFile':grid_file,'crossValidationFile':cv_file,
                      'fillValue':fixed_fill,'GeoCoordUnits':fixed_coord,'GeoCoordRefSystem':fixed_proj,
                      'XResolution':fixed_res,'YResolution':fixed_res,'ExtentXmin':xmin,
                      'ExtentXmax':xmax,'ExtentYmin':ymin,'ExtentYmax':ymax,'stationCount':str(nstns),
                      'gridPixCounties':npix,'rmseRH':rmse_fmt,'biasRH':bias_fmt,'maeRH':mae_fmt,'rsqRH':r2_fmt,
                      'credits':credit_statement,'contacts':contact_list}
    
    col1 = list(field_val_list.keys())
    col2 = list(field_val_list.values())
    meta_file_full = META_DIR + locality + meta_file
    with open(meta_file_full,'w') as fmeta:
        for key,val in zip(col1,col2):
            line = [key,val]
            fmt_line = "{:20}{:60}\n".format(*line)
            fmeta.write(fmt_line)
    return

def write_loocv(cv_stats,dt,icode):
    date_str = dt.strftime('%Y%m%d')
    if icode == 'state':
        locality = 'statewide/'
    else:
        locality = 'county/'+icode.upper()+'/'
    cv_file = CV_DIR + locality + '_'.join((date_str,'RH',icode,'loocv'))+'.csv'
    station_meta = MASTER_DF.set_index('SKN').loc[cv_stats.index]
    merged_df = station_meta.join(cv_stats,how='left')
    merged_df = merged_df.fillna('NA')
    merged_df.to_csv(cv_file)

def create_metadata(dt,icode):
    cv_stats = cross_validation(dt,icode)
    metrics = get_metrics(cv_stats)
    write_loocv(cv_stats,dt,icode)
    format_metadata(metrics,dt,icode)

if __name__=="__main__":
    icode = sys.argv[1]
    if len(sys.argv)>2:
        date_str = sys.argv[2]
        dt = pd.to_datetime(date_str)
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        dt = today - timedelta(days=1)
        dt = pd.to_datetime(dt.strftime('%Y-%m-%d'))

    cv_stats = cross_validation(dt,icode)
    metrics = get_metrics(cv_stats)
    write_loocv(cv_stats,dt,icode)
    format_metadata(metrics,dt,icode)