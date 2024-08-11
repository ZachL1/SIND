#!/bin/bash

# dataname="spaq"
dataname="para"
gpu_id=0

# methods=("CLIPIQA" "DBCNN" "HyperNet" "TOPIQ") # IQA
methods=("NIMA", "CLIPIQA", "TOPIQ") # IAA

for method in "${methods[@]}"; do
  src_yml="options/train/$method/leave_one_out_train_$method.yml"
  dst_dir=$(dirname "$src_yml")/temp
  mkdir -p "$dst_dir"

  for folder in data_json/for_leave_one_out/$dataname/*/; do
    echo "Processing $folder"
    if [ -d "$folder" ]; then
      folder_name=$(basename "$folder")
      train_json_path="$folder/train.json"
      test_json_path="$folder/test.json"

      exp_name="${method}_train_${dataname}_$folder_name"

      if [ -f "$train_json_path" ] && [ -f "$test_json_path" ]; then
        dst_yml="$dst_dir/$exp_name.yml"
        cp "$src_yml" "$dst_yml"

        yq eval ".datasets.train.meta_info_file = \"$train_json_path\"" -i "$dst_yml"
        yq eval ".datasets.val.meta_info_file = \"$test_json_path\"" -i "$dst_yml"
        yq eval ".name = \"$exp_name\"" -i "$dst_yml"
        yq eval ".datasets.train.name = \"$dataname\"" -i "$dst_yml"
        yq eval ".datasets.val.name = \"$dataname\"" -i "$dst_yml"
        if [[ "$method" != "NIMA" && ( "$dataname" == "spaq" || "$dataname" == "para" ) ]]; then
          yq eval ".datasets.val.augment.resize = 768" -i "$dst_yml"
        fi

        CUDA_VISIBLE_DEVICES=$gpu_id python pyiqa/train.py -opt $dst_yml
        
        echo "Generated $dst_yml with train and test paths for $folder_name."
      else
        echo "Warning: $train_json_path or $test_json_path does not exist."
      fi
    fi
  done
done
