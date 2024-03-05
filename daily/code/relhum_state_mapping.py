import sys
import pytz
import subprocess
import pandas as pd
from datetime import datetime,timedelta

#CONSTANTS---------------------------------------------------------------------
#MASTER_DIR = '/mnt/lts/nfs_fs02/hawaii_climate_risk_group/kodamak8/relhum/'
MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/relhum/'
COUNTY_MAP_DIR = MASTER_DIR + 'data_outputs/tiffs/daily/county/RH_map/'
STATE_MAP_DIR = MASTER_DIR + 'data_outputs/tiffs/daily/statewide/RH_map/'
#END CONSTANTS-----------------------------------------------------------------
#FUNCTIONS---------------------------------------------------------------------
def statewide_mosaic(varname,date_str,se_suffix=''):
    icode_list = ['bi','ka','mn','oa']
    date_tail = ''.join(date_str.split('-'))
    input_dir = COUNTY_MAP_DIR
    output_dir = STATE_MAP_DIR
    file_names = [input_dir+icode.upper()+'/'+'_'.join((varname,'map',icode.upper(),date_tail)) + se_suffix +'.tif' for icode in icode_list]
    output_name = output_dir + '_'.join((varname,'map','state',date_tail)) + se_suffix +'.tif'
    cmd = "gdal_merge.py -o "+output_name+" -of gtiff -co COMPRESS=LZW -n -9999 -a_nodata -9999"
    subprocess.call(cmd.split()+file_names)
#END FUNCTIONS-----------------------------------------------------------------

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
    try:
        statewide_mosaic('RH',date_str)
        print('Created state',date_str)
    except:
        print("Error on",date_str)