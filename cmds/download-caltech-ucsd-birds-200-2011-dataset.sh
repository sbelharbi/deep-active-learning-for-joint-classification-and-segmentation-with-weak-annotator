#!/usr/bin/env bash
# Script to download and extract the dataset: Caltech-UCSD-Birds-200-2011
# See: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

# cd to your folder where you want to save the data.
cd $1
mkdir Caltech-UCSD-Birds-200-2011
cd Caltech-UCSD-Birds-200-2011

# Download the images.
echo "Downloading images (1.1GB) ..."
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz

# Download masks (birds segmentations)
echo "Downloading segmentation (37MB) ..."
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz

# Downlaod the readme
echo "Downloading README.txt ..."
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/README.txt


echo "Finished downloading  Caltech-UCSD-Birds-200-2011 dataset."

echo "Extracting files ..."

tar -zxvf CUB_200_2011.tgz
tar -zxvf segmentations.tgz


echo "Finished extracting  Caltech-UCSD-Birds-200-2011 dataset."