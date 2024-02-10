#!/bin/bash

#"Fetching acquisition data and dependencies"
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/relhum_wget.py
#Processing acquisition data into aggregated daily data
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/hads_td_hour_parse.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/madis_td_hour_parse.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/dewpoint_aggregate_wrapper.py
#Prepping predictor data for mapping workflow
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/update_predictor_table.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/qaqc.py
#Running mapping and metadata workflow
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/mapping_wrapper.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/relhum_state_mapping.py
python3 -W ignore /home/hawaii_climate_products_container/preliminary/relhum/daily/code/meta_wrapper.py


