#!/bin/csh -f
##
## Wrapper script to convert VOC dataset to TFRecord
##--python3 object_detection/dataset_tools/create_pascal_tf_record.py \
##--  --label_map_path=object_detection/data/pascal_label_map.pbtxt \
##--  --data_dir=VOCdevkitTENNIS --year=TENNIS2019 --set=train \
##--  --output_path=data/TENNIS2019/tennis_train.record
##--
##--python3 object_detection/dataset_tools/create_pascal_tf_record.py \
##--  --label_map_path=object_detection/data/pascal_label_map.pbtxt \
##--  --data_dir=VOCdevkitTENNIS --year=TENNIS2019 --set=val \
##--  --output_path=data/TENNIS2019/tennis_val.record

set vocdir = `cd ../VOCdevkitMotion && pwd`
set bases = `find $vocdir -name 'ImageSets' | sed "s#$vocdir/\(.*\)/ImageSets#\1#" | sort`
echo "Following VOC datesets (or 'years') will be converted to TFRecord:"; echo $bases
set cdir = `pwd`
set map = "$cdir/tftraining/data/tennis_label_map_small.pbtxt"
set outdir = $cwd/data/TENNIS2019
if (! -d $outdir) mkdir $outdir

foreach base ( $bases )
  echo "--------------------------------------------------"
  echo $base
  @ cnt = `find $vocdir/$base/Annotations/ -name '*.xml' | wc -l`
  if ( $cnt < 2 ) then
    ## Skip if not enough annotations
    continue
  endif

  foreach set ( train val )
    python object_detection/dataset_tools/create_pascal_tf_record.py \
      --label_map_path=$map --data_dir=$vocdir --year=$base --set=${set} \
      --output_path=$outdir/tennis_${base}_${set}.record
  end
  
  echo ""

end

