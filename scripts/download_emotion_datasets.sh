mkdir -p data
wget -O data/hurricane.zip https://github.com/shreydesai/hurricane/archive/refs/heads/master.zip
unzip data/hurricane.zip -d data/
mv data/hurricane-master data/hurricane
rm data/hurricane.zip

cd data
# if you prefer not to use svn, please visit https://github.com/google-research/google-research/tree/master/goemotions and download the files directly
svn export https://github.com/google-research/google-research/trunk/goemotions