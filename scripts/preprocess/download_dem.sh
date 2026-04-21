#!/bin/bash
set -euo pipefail

CONFIG=$1

# Extract target path directly from config using Python to avoid adding yq as a dependency
OUTPUT_FILE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['STATIC_DEM_PATH'])")
mkdir -p "$(dirname "$OUTPUT_FILE")"

TARGET_PROJ="+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
BBOX="-10.43 31.75 57.81 67.62"
TARGET_RES="2000"
VRT_URL="/vsicurl/https://opentopography.s3.sdsc.edu/raster/COP30/COP30_hh.vrt"

echo "Initiating retrieval and reprojection of digital elevation model..."
gdalwarp -t_srs "$TARGET_PROJ" \
         -te_srs EPSG:4326 -te $BBOX \
         -tr $TARGET_RES $TARGET_RES \
         -r bilinear \
         -wm 2048 \
         -multi \
         -co COMPRESS=DEFLATE \
         -co TILED=YES \
         "$VRT_URL" "$OUTPUT_FILE"