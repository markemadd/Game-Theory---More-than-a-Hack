#!/usr/bin/env bash
# Download the seven open IDSSE matches (Bassek et al. 2025, CC-BY 4.0) into ext_data/idsse/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p ext_data/idsse
BASE=https://huggingface.co/datasets/pysport/idsse-data/resolve/main
for m in 000001_DFL-MAT-J03WMX 000001_DFL-MAT-J03WN1 000002_DFL-MAT-J03WPY 000002_DFL-MAT-J03WOH \
         000002_DFL-MAT-J03WQQ 000002_DFL-MAT-J03WOY 000002_DFL-MAT-J03WR9; do
  id=${m#*MAT-}
  curl -sfL -o "ext_data/idsse/meta_${id}.xml" "$BASE/DFL_02_01_matchinformation_DFL-COM-$m.xml"
  curl -sfL -o "ext_data/idsse/pos_${id}.xml" "$BASE/DFL_04_03_positions_raw_observed_DFL-COM-$m.xml"
done
ls -la ext_data/idsse
