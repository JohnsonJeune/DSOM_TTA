target=(1 2 3 4 5)

for target_index in ${target[@]}
    do
    python adapt.py \
    --target_domain $target_index \
    --dataset_cfg './cfg/dataset/pamap2.yaml' \
    --algorithm_cfg './cfg/algorithm/tent.yaml'
done

target=(3 4)

for target_index in ${target[@]}
    do
    python adapt.py \
    --target_domain $target_index \
    --dataset_cfg './cfg/dataset/pamap2.yaml' \
    --algorithm_cfg './cfg/algorithm/tent.yaml'
done