import sys
import pytz
import pandas as pd
from datetime import datetime,timedelta
from meta_data import create_metadata

ICODE_LIST = ['BI','KA','MN','OA','state']

if __name__=="__main__":
    if len(sys.argv)>1:
        date_str = sys.argv[1]
        dt = pd.to_datetime(date_str)
    else:
        hst = pytz.timezone('HST')
        today = datetime.today().astimezone(hst)
        dt = today - timedelta(days=1)
        dt = pd.to_datetime(dt.strftime('%Y-%m-%d'))
    
    print(dt.strftime('%Y-%m-%d'),'metadata')
    for icode in ICODE_LIST:
        try:
            create_metadata(dt,icode)
            print(icode,'succeeded')
        except:
            print(icode,'failed on',dt.strftime('%Y-%m-%d'))
