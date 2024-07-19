import sys
import pytz
import pandas as pd
from relhum_daily_mapping import map_main,backup_main,split_dataframe
from datetime import datetime,timedelta

#MASTER_DIR = '/home/kodamak8/nrt_testing/relhum_test/'
MASTER_DIR = '/home/hawaii_climate_products_container/preliminary/'
ICODE_LIST = ['BI','MN','KA','OA']
TD_DIR = MASTER_DIR + 'relhum/data_outputs/tables/station_data/daily/partial_filled/statewide/'
MASTER_LINK = 'https://raw.githubusercontent.com/ikewai/hawaii_wx_station_mgmt_container/main/Hawaii_Master_Station_Meta.csv'
MASTER_DF = pd.read_csv(MASTER_LINK)
META_COLS = list(MASTER_DF.set_index('SKN').columns)

if __name__=="__main__":
    if len(sys.argv)>1:
        date_input = sys.argv[1]
        dt = pd.to_datetime(date_input)
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        dt = today - timedelta(days=1)
        dt = pd.to_datetime(dt.strftime('%Y-%m-%d'))

    year_mon = dt.strftime('%Y_%m')
    td_file = TD_DIR + '_'.join(('daily','TD',year_mon,'qc'))+'.csv'
    td_df = pd.read_csv(td_file)
    td_data,td_meta = split_dataframe(td_df,META_COLS)
    
    if dt in list(td_data.columns):
        td_day = td_data[dt].dropna()
        print('at ',dt)
        for icode in ICODE_LIST:
            map_main(dt,td_day,icode)
        """try:
            if td_day.shape[0]>=10:
                for icode in ICODE_LIST:
                    map_main(dt,td_day,icode)
                    print("Success -- Created RH map for",icode,dt.strftime('%Y-%m-%d'))
            else:
                for icode in ICODE_LIST:
                    backup_main(dt,icode)
                    print("Success -- Backup RH map created for",icode,dt.strftime('%Y-%m-%d'))
            
        except:
            print("Error. RH map failed on",dt.strftime('%Y-%m-%d'))"""

