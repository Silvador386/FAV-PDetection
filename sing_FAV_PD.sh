#!/bin/bash
#PBS -q iti
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_ssd=75gb:gpu_cap=cuda75
#PBS -j oe
#PBS -o /storage/plzen1/home/silvador386/
#PBS -m ae
#trap 'clean_scratch' TERM EXIT


sing_image=mmdetect_img.sif


# -- tested by:
##$ qsub -I -l select=1:ncpus=1:ngpus=1:mem=16gb:scratch_ssd=75gb:gpu_cap=cuda75 -l walltime=1:00:00 -q gpu

cp -r /storage/plzen1/home/silvador386/FAV-PDetection/ "$SCRATCHDIR" || exit $LINENO

WORK_PATH=$SCRATCHDIR/FAV-Detection/code/
DATA_PATH=$SCRATCHDIR/FAV-PDetection/data/P-DESTRE/coco_format/
OUTPUT_PATH=$SCRATCHDIR/FAV-PDetection/work_dirs/

cd "$WORK_PATH" || exit $LINENO



today=$(date +%Y%m%d%H%M)
singularity exec --nv -B "$SCRATCHDIR"  ../"$sing_image" \
  python main.py > "$OUTPUT_PATH"/"$today"_fav_pd.log

mkdir /storage/plzen1/home/silvador386/FAV-PDetection_Output/"$today"
cp -r "$OUTPUT_PATH" /storage/plzen1/home/silvador386/FAV-PDetection_Output/"$today"
cp ../"$WORK_PATH"/sing_FAV_PD.sh /storage/plzen1/home/silvador386/FAV-PDetection_Output/"$today"

