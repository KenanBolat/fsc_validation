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
  f_name=$(echo $file | cut -c 1-41)
  echo $f_name
  f_pat="${f_name}.tif"
  temp_file="${f_name}_temp.tif"

  # Remove temporary files
  rm -f $f_pat
  rm -f $temp_file

  # convert hdf to geotiff
  gdal_translate HDF4_EOS:EOS_GRID:"${file}":MOD_Grid_Snow_500m:NDSI_Snow_Cover $temp_file

  # reproject to WGS84
  gdalwarp -of GTIFF -s_srs '+proj=sinu +R=6371007.181 +nadgrids=@null +wktext' -t_srs '+proj=longlat +datum=WGS84 +no_defs' $temp_file $f_pat
  end=$(date +%s)
  runtime=$((end - start))
  echo "$file has been exhausted within $runtime "
  rm -f $temp_file
fi
