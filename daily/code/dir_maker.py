import os

#MASTER_DIR = '/mnt/lustre/koa/koastore/hawaii_climate_risk_group/kodamak8/relhum/'
MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/relhum/'
code_dir = MASTER_DIR + 'daily/code/'
dep_dir = MASTER_DIR + 'daily/dependencies/'
output_dir = MASTER_DIR + 'data_outputs/'

work_dir = MASTER_DIR + 'working_data/'
meta_dir = output_dir + 'metadata/'
tiff_dir = output_dir + 'tiffs/'
tab_dir = output_dir + 'tables/'

#make data aqs dir
os.makedirs('/home/hawaii_climate_products_container/preliminary/data_aqs/data_outputs/',exist_ok=True)
os.makedirs('/home/hawaii_climate_products_container/preliminary/data_aqs/data_outputs/madis/parse/',exist_ok=True)
os.makedirs('/home/hawaii_climate_products_container/preliminary/data_aqs/data_outputs/hads/parse/',exist_ok=True)
os.makedirs(MASTER_DIR,exist_ok=True)
os.makedirs(code_dir,exist_ok=True)
os.makedirs(dep_dir,exist_ok=True)
os.makedirs(output_dir,exist_ok=True)
os.makedirs(meta_dir,exist_ok=True)
os.makedirs(tiff_dir,exist_ok=True)
os.makedirs(tab_dir,exist_ok=True)
os.makedirs(work_dir,exist_ok=True)
os.makedirs(work_dir+'TA_record/',exist_ok=True)
os.makedirs(dep_dir+'geoTiffs_250m/',exist_ok=True)
os.makedirs(dep_dir+'geoTiffs_250m/dem/',exist_ok=True)
os.makedirs(dep_dir+'predictors/',exist_ok=True)
os.makedirs(dep_dir+'pickle/',exist_ok=True)
os.makedirs(dep_dir+'gapfill/',exist_ok=True)
os.makedirs(dep_dir+'gridded_temp/',exist_ok=True)
os.makedirs(work_dir+'processed_data/',exist_ok=True)
os.makedirs(work_dir+'processed_data/hads/',exist_ok=True)
os.makedirs(work_dir+'processed_data/madis/',exist_ok=True)
os.makedirs(work_dir+'processed_data/hiMeso/',exist_ok=True)
os.makedirs(meta_dir+'daily/',exist_ok=True)
os.makedirs(meta_dir+'daily/county/',exist_ok=True)
os.makedirs(meta_dir+'daily/statewide/',exist_ok=True)
os.makedirs(meta_dir+'daily/county/BI/',exist_ok=True)
os.makedirs(meta_dir+'daily/county/KA/',exist_ok=True)
os.makedirs(meta_dir+'daily/county/MN/',exist_ok=True)
os.makedirs(meta_dir+'daily/county/OA/',exist_ok=True)

os.makedirs(tab_dir+'relhum_station_tracking/',exist_ok=True)
os.makedirs(tab_dir+'loocv/',exist_ok=True)
os.makedirs(tab_dir+'station_data/',exist_ok=True)
os.makedirs(tab_dir+'loocv/daily/',exist_ok=True)
os.makedirs(tab_dir+'loocv/daily/county/',exist_ok=True)
os.makedirs(tab_dir+'loocv/daily/statewide/',exist_ok=True)
os.makedirs(tab_dir+'loocv/daily/county/BI/',exist_ok=True)
os.makedirs(tab_dir+'loocv/daily/county/KA/',exist_ok=True)
os.makedirs(tab_dir+'loocv/daily/county/MN/',exist_ok=True)
os.makedirs(tab_dir+'loocv/daily/county/OA/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/raw/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/raw_qc/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/raw/county/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/raw/statewide/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/raw_qc/county/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/raw_qc/statewide/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/partial_filled/statewide/',exist_ok=True)
os.makedirs(tab_dir+'station_data/daily/qc_flags/',exist_ok=True)

os.makedirs(tiff_dir+'daily/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/',exist_ok=True)
os.makedirs(tiff_dir+'daily/statewide/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_map/',exist_ok=True)
#os.makedirs(tiff_dir+'daily/county/RH_se/',exist_ok=True)
os.makedirs(tiff_dir+'daily/statewide/RH_map/',exist_ok=True)
#os.makedirs(tiff_dir+'daily/statewide/RH_se/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_map/BI/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_map/KA/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_map/MN/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_map/OA/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_se/BI/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_se/KA/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_se/MN/',exist_ok=True)
os.makedirs(tiff_dir+'daily/county/RH_se/OA/',exist_ok=True)
