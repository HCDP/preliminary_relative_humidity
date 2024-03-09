# Start with the Relative Humidity Preliminary Aggregation Base.
FROM ghcr.io/hcdp/task-preliminary-relative-humidity-base:latest
LABEL org.opencontainers.image.source="https://github.com/hcdp/preliminary_relative_humidity"
LABEL org.opencontainers.image.description="Daily Relative Humidity Aggregation and Mapping."

RUN mkdir -p /home/hawaii_climate_products_container/preliminary/relhum/daily
ADD /daily /home/hawaii_climate_products_container/preliminary/relhum/daily

# Add the upload config
ADD /containers/daily/configs/upload/upload_config.json /sync/upload_config.json

# Add the task wrapper script.
ADD /containers/daily/task.sh /actor/task.sh

# This container is intended to run the task wrapper at launch, unless otherwise specified at run-time.
CMD [ "/bin/bash", "/actor/task.sh" ]