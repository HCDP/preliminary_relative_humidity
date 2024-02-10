import sys
import subprocess
import pytz
import pandas as pd
from datetime import datetime, timedelta

PARENT_DIR = r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/workflow_data/preliminary_test/'
REMOTE_BASEURL =r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/production/temperature/'
REMOTE_DEPEND = r'https://ikeauth.its.hawaii.edu/files/v2/download/public/system/ikewai-annotated-data/HCDP/temperature/'
LOCAL_PARENT = r'/home/hawaii_climate_products_container/preliminary/'
#LOCAL_PARENT = r'/home/kodamak8/nrt_testing/relhum_test/'
LOCAL_DATA_AQS = LOCAL_PARENT + r'relhum/working_data/data_aqs/'
#LOCAL_DATA_AQS = LOCAL_PARENT + r'scratch/'
LOCAL_DEPEND = LOCAL_PARENT + r'relhum/daily/'
LOCAL_TEMP = LOCAL_DEPEND + r'dependencies/gridded_temp/'
SRC_LIST = ['hads','madis']
ICODE_LIST = ['BI','KA','MN','OA','state']

if __name__=='__main__':
    if len(sys.argv) > 1:
        input_date = sys.argv[1]
        dt = pd.to_datetime(input_date)
        prev_day_day = dt.strftime('%Y%m%d')
        prev_day_mon = dt.strftime('%Y_%m')
        year_str = dt.strftime('%Y')
        mon_str = dt.strftime('%m')
        ymd_str = dt.strftime('%Y_%m_%d')
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        prev_day = today - timedelta(days=1)
        prev_day_day = prev_day.strftime('%Y%m%d')
        prev_day_mon = prev_day.strftime('%Y_%m')
        year_str = prev_day.strftime('%Y')
        mon_str = prev_day.strftime('%m')
        ymd_str = prev_day.strftime('%Y_%m_%d')

    #Pull the data aq files
    for src in SRC_LIST:
        src_url = PARENT_DIR+r'data_aqs/data_outputs/'+src+r'/parse/'
        dest_url = LOCAL_DATA_AQS + src + r'/parse/'
        filename = src_url + r'_'.join((prev_day_day,src,'parsed')) + r'.csv'
        local_name = dest_url + r'_'.join((prev_day_day,src,'parsed')) + r'.csv'
        cmd = ["wget",filename,"-O",local_name]
        subprocess.call(cmd)

    
    #Pull dependencies
    depend_src = REMOTE_DEPEND + r'td_dependencies.tar.gz'
    dest_path = LOCAL_DEPEND + r'td_dependencies.tar.gz'
    cmd = ["wget",depend_src,"-O",dest_path]
    subprocess.call(cmd)
    cmd = ["tar","-xvf",dest_path,"-C",LOCAL_DEPEND]
    subprocess.call(cmd)

    #Pull in the temperature maps
    for icode in ICODE_LIST:
        if icode == 'state':
            locality = 'statewide'
        else:
            locality = icode.lower()
        max_src_path = REMOTE_BASEURL + r'max/day/'+locality+r'/data_map/'+year_str+r'/'+mon_str+r'/'
        max_src_file = max_src_path + '_'.join(('temperature','max','day',locality,'data_map',ymd_str))+'.tif'
        max_dest_file = LOCAL_TEMP + '_'.join(('Tmax','map',icode,prev_day_day))+'.tif'
        min_src_path = REMOTE_BASEURL + r'min/day/'+locality+r'/data_map/'+year_str+r'/'+mon_str+r'/'
        min_src_file = min_src_path + '_'.join(('temperature','min','day',locality,'data_map',ymd_str))+'.tif'
        min_dest_file = LOCAL_TEMP + '_'.join(('Tmin','map',icode,prev_day_day))+'.tif'
        cmd = ["wget",max_src_file,"-O",max_dest_file]
        subprocess.call(cmd)
        cmd = ["wget",min_src_file,"-O",min_dest_file]
        subprocess.call(cmd)