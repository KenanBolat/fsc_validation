#!/usr/bin/bash
while getopts d:f: flag; do
  case "${flag}" in
  d) date_=${OPTARG} ;;
  f) file=${OPTARG} ;;
  *) file="" ;;
  esac
done


if [[ -z "$file" ]];
 then
  echo "NO file provided"
else
  start=$(date +%s)

  # cd /media/knn/SSD/modis_hdf/all_hdf
  # cd /media/knn/SSD/modis_hdf/all_hdf/tif
  echo "$file is being done"
  f_name=$(echo $file | cut -c 1-23)
  echo $f_name
  f_pat="${f_name}_resampled.tif"
  temp_file="${f_name}_temp.tif"

  # Remove temporary files
  rm -f $f_pat
  rm -f $temp_file

  # resample geotiff to mars
  gdalwarp -tr 0.01 0.01 -r near -te -179.995 0.005000000000009663 179.995 89.995 -of GTiff $file $f_pat
  end=$(date +%s)
  runtime=$((end - start))
  echo "$file has been exhausted within $runtime "
  rm -f $temp_file
fi
