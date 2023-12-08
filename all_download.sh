#./bash/clean_downloads.sh
#./bash/download_body_models.sh
#./bash/download_misc.sh
#./bash/download_cropped_images.sh
#./bash/download_splits.sh
#./bash/download_feat.sh
#./bash/download_baselines.sh
#./bash/download_images.sh
python scripts_data/checksum.py # verify checksums; this could take a while
python scripts_data/unzip_download.py # unzip downloaded 
