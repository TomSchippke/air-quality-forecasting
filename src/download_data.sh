#!/bin/bash
# Download raw data
mkdir -p data/raw

wget https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip -O data/raw/beijing_air.zip

# Unzip the data
unzip data/raw/PRSA2017_Data_20130301-20170228.zip -d data/raw/

# Remove the zip file
rm data/raw/data.csv data/raw/test.csv data/raw/*.JPG data/raw/beijing_air.zip data/raw/PRSA2017_Data_20130301-20170228.zip
