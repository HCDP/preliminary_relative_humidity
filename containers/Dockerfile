FROM ghcr.io/hcdp/task-base:latest
LABEL org.opencontainers.image.source="https://github.com/hcdp/preliminary_relative_humidity"
LABEL org.opencontainers.image.description="Base Image for preliminary relative humidity workflows."

## PACKAGE/LIBRARY INSTALLATIONS ##
# Fetch the latest apt repo information.
RUN apt update
# Let APT know that we don't have a terminal.
ENV DEBIAN_FRONTEND=noninteractive

# Upgrade pip for rpy2 install
RUN pip install --upgrade pip

# Install libgdal
# Check version
RUN apt-get install -y libgdal-dev

#Install python setuptools for rpy2 install
RUN apt-get install -y python3-setuptools

# Time zone package
RUN apt-get install -y tzdata

#Install R
RUN apt-get install -y r-base
RUN R -e 'install.packages("earth", version="5.3-2", repos="https://ftp.ussg.iu.edu/CRAN/")'

# Install RH python dependencies
RUN python3 -m pip install numpy
RUN python3 -m pip install scipy
RUN python3 -m pip install gdal==$(gdal-config --version)
RUN python3 -m pip install rasterio
RUN python3 -m pip install matplotlib
RUN python3 -m pip install pandas
RUN python3 -m pip install rpy2
RUN python3 -m pip install affine
RUN python3 -m pip install pyproj
RUN python3 -m pip install statsmodels
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install tiffile
RUN python3 -m pip install python-dotenv
RUN python3 -m pip install pathlib

RUN mkdir -p /sync
RUN wget https://raw.githubusercontent.com/ikewai/data-uploader/main/upload.py -O /sync/upload.py

RUN mkdir -p /home/hawaii_climate_products_container/preliminary/relhum/data_outputs
ADD /data_outputs /home/hawaii_climate_products_container/preliminary/relhum/data_outputs

RUN mkdir -p /home/hawaii_climate_products_container/preliminary/relhum/working_data
ADD /working_data /home/hawaii_climate_products_container/preliminary/relhum/working_data

ADD /tools /sync
