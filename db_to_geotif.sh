#!/bin/bash
#db information
hostname=192.168.0.221
username=postgres
pass=$COMMON_PASSWORD
while read p; do
  echo "$p"
  shp_name="${p}.shp"
  tif_name="${p}.tif"
  pgsql2shp -f $shp_name -h $hostname -u postgres -P $pass MGM "select snow_depth, geom from validation_2020 where status <> 0  and m_date =  '${p}'"
  gdal_rasterize -l $p -a SNOW_DEPTH -ts 35999.0 17998.0 -init 0.0 -a_nodata 0.0 -te -179.99 -89.99 179.99 89.99 -a_srs epsg:4326 -ot UInt16 -of GTiff $shp_name $tif_name
  break
done < dates.txt
