#!/bin/bash

#!/bin/bash
echo "[task.sh] [1/7] Starting Execution."
export TZ="HST"
echo "It is currently $(date)."
if [ $AGGREGATION_DATE ]; then
    echo "An aggregation date was provided by the environment."
    echo "Aggregation date is: " $AGGREGATION_DATE
else
    export AGGREGATION_DATE=$(date --iso-8601)
    echo "No aggregation date was provided by the environment. Defaulting to today."
    echo "Aggregation date is: " $AGGREGATION_DATE
fi
export AGGREGATION_DATE_YESTERDAY=$(date --date="$AGGREGATION_DATE - 1 day" --iso-8601)
echo "Yesterday is: " $AGGREGATION_DATE_YESTERDAY

echo "[task.sh] [2/7] Fetching acquisition data and dependencies."
#"Fetching acquisition data and dependencies"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/relhum_wget.py

echo "[task.sh] [3/7] Processing acquisition data into aggregated daily data."
#Processing acquisition data into aggregated daily data
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/hads_td_hour_parse.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/madis_td_hour_parse.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/dewpoint_aggregate_wrapper.py

echo "[task.sh] [4/7] Prepping predictor data for mapping workflow."
#Prepping predictor data for mapping workflow
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/update_predictor_table.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/qaqc.py

echo "[task.sh] [5/7] Running mapping and metadata workflow."
#Running mapping and metadata workflow
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/mapping_wrapper.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/relhum_state_mapping.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/meta_wrapper.py

echo "[task.sh] [6/7] Preparing upload config."
cd /sync
python3 update_date_string_in_config.py upload_config.json upload_config_datestrings_loaded.json $AGGREGATION_DATE
python3 add_upload_list_to_config.py upload_config_datestrings_loaded.json config.json
python3 add_auth_info_to_config.py config.json

echo "[task.sh] [7/7] Uploading data."
python3 upload.py

echo "[task.sh] All done!"