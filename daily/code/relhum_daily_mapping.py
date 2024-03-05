import sys
import warnings
import rasterio
import pickle
import pandas as pd
import numpy as np
from osgeo import gdal
from affine import Affine
from pyproj import Transformer
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri

#CONSTANTS--------------------------------------------------------------------
earth = rpackages.importr("earth")
#MASTER_DIR = '/home/kodamak8/nrt_testing/relhum_test/'
MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/'
DEP_DIR = MASTER_DIR + 'relhum/daily/dependencies/'
PICK_DIR = DEP_DIR + 'pickle/'
ELEV_DIR = DEP_DIR + '/geoTiffs_250m/dem/'
TD_DIR = MASTER_DIR + 'relhum/data_outputs/tables/station_data/daily/partial_filled/statewide/'
PRED_DIR = DEP_DIR + 'predictors/'
GAPFILL_DIR = DEP_DIR + 'gapfill/'
TMINMAX_DIR = DEP_DIR + 'gridded_temp/'
RH_MAP_DIR = MASTER_DIR + 'relhum/data_outputs/tiffs/daily/county/RH_map/'
MASTER_LINK = 'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
MASTER_DF = pd.read_csv(MASTER_LINK)
META_COLS = list(MASTER_DF.set_index('SKN').columns)
PRED_LIST = ['dem_250']
GAPFILL_STNS = {339.5:[107.4,278,279,339.4],339.6:[278,279,339.5]}
INVERSION = 2100
NO_DATA_VAL = -9999
#END CONSTANTS----------------------------------------------------------------
#FUNCTIONS--------------------------------------------------------------------
def load_pickle(pick_name):
    with open(pick_name,'rb') as f:
        pick_models = pickle.load(f)
    return pick_models

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

def linear(x,a,b):
    return a*x+b

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

def get_coordinates(GeoTiff_name):

    # Read raster
    with rasterio.open(GeoTiff_name) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        A = r.read()  # pixel values

    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing
    # at centre
    def rc2en(r, c): return T1 * (c, r)

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(
        rc2en, otypes=[
            float, float])(
        rows, cols)

    transformer = Transformer.from_proj(
        'EPSG:4326',
        '+proj=longlat +datum=WGS84 +no_defs +type=crs',
        always_xy=True)

    LON, LAT = transformer.transform(eastings, northings)
    return LON, LAT

def get_data_array(tiffname):

    raster = rasterio.open(tiffname)

    raster_data = raster.read(1)
    raster_mask = raster.read_masks(1)

    raster_mask[raster_mask > 0] = 1
    masked_array = raster_data * raster_mask

    masked_array[raster_mask == 0] = np.nan
    shape = masked_array.shape

    return masked_array, raster_mask, shape

def output_tiff(df_data,base_tiff,tiff_filename,shape):
    cols, rows = shape
    
    ds = gdal.Open(base_tiff)

    # arr_out = np.where((arr < arr_mean), -100000, arr)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(tiff_filename, rows, cols, 1, gdal.GDT_Float32)
    # sets same geotransform as input
    outdata.SetGeoTransform(ds.GetGeoTransform())
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(df_data.reshape(shape))
    # if you want these values (in the mask) transparent
    outdata.GetRasterBand(1).SetNoDataValue(NO_DATA_VAL)
    outdata.FlushCache()  # saves to disk!!
    outdata = None
    ds = None

def map_main(dt,td_day,icode):
    #pred_file = PRED_DIR + 'upd_predictors.csv'
    pred_file = PRED_DIR + 'updated_TD_predictors.csv'
    pick_file = PICK_DIR + 'ta_model_pickle.dat'
    pick_model = load_pickle(pick_file)
    ta_theta = pick_model['thetas'][0]
    pred_df = pd.read_csv(pred_file)
    pred_df = pred_df.set_index('SKN')
    #Pred day should not have any missing values, but prevents erroring out in case there are
    pred_day = pred_df.loc[td_day.index,PRED_LIST].dropna()
    match_skns = np.intersect1d(pred_day.index.values,td_day.index.values)
    match_preds = pred_day.loc[match_skns]
    match_td = td_day.loc[match_skns]
    #Get appropriate Tmax map and estimate as 1pm TA
    date_code = dt.strftime('%Y%m%d')
    #tmax_file = TMINMAX_DIR + icode.upper()+ '/' + '_'.join(('temperature_max_day',icode.lower(),'data_map',dt.strftime('%Y_%m_%d')))+'.tif'
    tmax_file = TMINMAX_DIR + '_'.join(('Tmax','map',icode,date_code))+'.tif'
    tmax_data,tmax_mask,tmax_shape = get_data_array(tmax_file)
    tmax1d = tmax_data.reshape(-1)
    ta_est = linear(tmax1d,*ta_theta)
    ta_grid = ta_est.reshape(tmax_shape)
    #Get dem data
    dem_file = ELEV_DIR + '_'.join((icode.lower(),'dem_250m.tif'))
    dem_data,dem_mask,dem_shape = get_data_array(dem_file)
    dem1d = dem_data.reshape(-1)
    td_est = earth_to_mars(match_preds.values,match_td.values,dem1d,pred_list=PRED_LIST)
    td_grid = td_est.reshape(dem_shape)
    vp = vapor_pressure(td_grid)
    es = vapor_pressure(ta_grid)
    rh_grid = compute_rh(vp,es)
    rh_grid[np.isnan(rh_grid)] = NO_DATA_VAL
    rh_file = RH_MAP_DIR + icode.upper() + '/' + '_'.join(('RH','map',icode,date_code))+'.tif'
    output_tiff(rh_grid,dem_file,rh_file,dem_shape)

def backup_main(dt,icode):
    backup_file = PICK_DIR + 'td_tmin_model_v1.dat'
    backup_model = load_pickle(backup_file)
    lower_model = backup_model['thetas'][0]
    upper_model = backup_model['thetas'][1]
    ta_file = PICK_DIR + 'ta_model_pickle.dat'
    ta_pickle = load_pickle(ta_file)
    ta_model = ta_pickle['thetas'][0]
    date_code = dt.strftime('%Y%m%d')
    tmin_file = TMINMAX_DIR + '_'.join(('Tmin','map',icode,date_code))+'.tif'
    tmin_data,tmin_mask,tmin_shape = get_data_array(tmin_file)
    tmin1d = tmin_data.reshape(-1)
    dem_file = ELEV_DIR + '_'.join((icode.lower(),'dem_250m.tif'))
    dem_data,dem_mask,dem_shape = get_data_array(dem_file)
    dem1d = dem_data.reshape(-1)
    tmax_file = TMINMAX_DIR + '_'.join(('Tmax','map',icode,date_code))+'.tif'
    tmax_data,tmax_mask,tmax_shape = get_data_array(tmax_file)
    tmax1d = tmax_data.reshape(-1)
    ta_est = linear(tmax1d,*ta_model)
    ta_grid = ta_est.reshape(tmax_shape)
    Zlow = np.where(dem1d<=INVERSION)
    Zhigh = np.where(dem1d>INVERSION)
    tmin_high = tmin1d[Zhigh]
    td_est = linear(tmin1d,*lower_model)
    td_est[Zhigh] = linear(tmin1d[Zhigh],*upper_model)
    td_grid = td_est.reshape(tmin_shape)
    vp = vapor_pressure(td_grid)
    es = vapor_pressure(ta_grid)
    rh_grid = compute_rh(vp,es)
    rh_grid[np.isnan(rh_grid)] = NO_DATA_VAL
    rh_file = RH_MAP_DIR + icode.upper() + '/' + '_'.join(('RH','map',icode,date_code))+'.tif'
    output_tiff(rh_grid,dem_file,rh_file,dem_shape)


#END FUNCTIONS----------------------------------------------------------------

if __name__=="__main__":
    icode = sys.argv[1]
    date_range = sys.argv[2]
    st_date = pd.to_datetime(date_range.split('-')[0])
    en_date = pd.to_datetime(date_range.split('-')[1])
    date_iter = pd.date_range(st_date,en_date)

    for dt in date_iter:
        year_mon = dt.strftime('%Y_%m')
        td_file = TD_DIR + '_'.join(('daily','TD',year_mon,'qc'))+'.csv'
        td_df = pd.read_csv(td_file)
        td_data,td_meta = split_dataframe(td_df,META_COLS)
        try:
            td_day = td_data[dt].dropna()
            if td_day.shape[0] >= 10:
                map_main(dt,td_day,icode)
            else:
                print('Insufficient data points. Revert to backup.')
                backup_main(dt,icode)
            print(dt.strftime('%Y-%m-%d'),'- Success')
        except:
            #Create backup map based on non-daily dependent data
            backup_main(dt,icode)
            print('No data on',dt.strftime('%Y-%m-%d'),'- Used backup.')
