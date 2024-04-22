#!/bin/bash

#!/bin/bash
echo "[task.sh] [1/7] Starting Execution."
export TZ="HST"
echo "It is currently $(date)."
if [ $CUSTOM_DATE ]; then
    echo "An aggregation date was provided by the environment."
else
    export CUSTOM_DATE=$(date -d "1 day ago" --iso-8601)
    echo "No aggregation date was provided by the environment. Defaulting to yesterday."
fi
echo "Aggregation date is: " $CUSTOM_DATE

echo "[task.sh] [2/7] Fetching acquisition data and dependencies."
#"Fetching acquisition data and dependencies"
echo "---relhum_wget.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/relhum_wget.py $CUSTOM_DATE

echo "[task.sh] [3/7] Processing acquisition data into aggregated daily data."
#Processing acquisition data into aggregated daily data
echo "---hads_td_hour_parse.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/hads_td_hour_parse.py $CUSTOM_DATE
echo "---madis_td_hour_parse.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/madis_td_hour_parse.py $CUSTOM_DATE
echo "---dewpoint_aggregate_wrapper.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/dewpoint_aggregate_wrapper.py $CUSTOM_DATE

echo "[task.sh] [4/7] Prepping predictor data for mapping workflow."
#Prepping predictor data for mapping workflow
echo "---update_predictor_table.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/update_predictor_table.py $CUSTOM_DATE
echo "---qaqc.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/qaqc.py $CUSTOM_DATE

echo "[task.sh] [5/7] Running mapping and metadata workflow."
#Running mapping and metadata workflow
echo "---mapping_wrapper.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/mapping_wrapper.py $CUSTOM_DATE
echo "---relhum_state_mapping.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/relhum_state_mapping.py $CUSTOM_DATE
echo "---meta_wrapper.py---"
python3.8 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/meta_wrapper.py $CUSTOM_DATE

echo "[task.sh] [6/7] Preparing upload config."
cd /sync
python3 update_date_string_in_config.py upload_config.json upload_config_datestrings_loaded.json $CUSTOM_DATE
python3 add_upload_list_to_config.py upload_config_datestrings_loaded.json config.json
python3 add_auth_info_to_config.py config.json

echo "[task.sh] [7/7] Uploading data."
python3 upload.py

echo "[task.sh] All done!"