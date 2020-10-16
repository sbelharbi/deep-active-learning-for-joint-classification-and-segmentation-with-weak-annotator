#!/usr/bin/env bash
# Script to download and extract the dataset: Glas. (GlaS-2015)
# See: https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/

# cd to your folder where you want to save the data.
cd $1
mkdir GlaS-2015
cd GlaS-2015

# Download the images.
echo "Downloading  ..."
wget https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip

echo "Finished downloading  GlaS-2015."

echo "Extracting files ..."

unzip  warwick_qu_dataset_released_2016_07_08.zip

echo "========================================================================="
echo "                 We are aware of the SPACES in the folder name. "
echo "            Not sure whose idea was to put space in the folder name. "
echo "                         I let you to deal with it. "
echo "========================================================================="


echo "Finished extracting  GlaS-2015."