#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh

# Initialize variables with default values
ckpts=""
config=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpts)
            ckpts="$2"
            shift 2
            ;;
        --config)
            config="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --ckpts <path> --config <path>"
            exit 1
            ;;
    esac
done

if [[ -z "$ckpts" || -z "$config" ]]; then
    echo "Missing required arguments."
    echo "Usage: $0 --ckpts <path> --config <path>"
    exit 1
fi

conda activate TractoBERT

# Path to your config file read from --config argument yaml file
dataset_config_file=$(python -c "
import yaml
with open('$config', 'r') as f:
    config = yaml.safe_load(f)
print(config['dataset']['test']['_base_'])
")

echo "dataset_config_file: $dataset_config_file"
sleep 2

# Extract dataset name, model and mapping path from the config file
dataset_name=$(python -c "
from utils.config import cfg_from_yaml_file
config = cfg_from_yaml_file('$dataset_config_file')
print(config.NAME)
")
echo "dataset_name: $dataset_name"
sleep 2
model=$(python -c "
model = '$config'.split('/')[-2]
print(model)
")
echo "model: $model"
sleep 2
mapping_path=$(python -c "
from utils.config import cfg_from_yaml_file
config = cfg_from_yaml_file('$config')
print(config.dataset.test._base_.label_to_int_mapping_path)
")
echo "mapping_path: $mapping_path"
sleep 2

# Extract exp_name and exp_subset from ckpts path
exp_name=$(python -c "
exp_name = '$ckpts'.split('/')[-2]
print(exp_name)
")
echo "exp_name: $exp_name"
sleep 2
exp_subset=$(python -c "
pth = '$ckpts'.split('experiments')[-1]
if len(pth) >=2:
    print(pth.split('/')[1])
else:
    print('/')
")
echo "exp_subset: $exp_subset"
sleep 2


# Define the subject_ids blocks (pairs of subject_ids lines to toggle between)
subject_blocks=(
    "test_subjects: ['sub-1175', 'sub-1046', 'sub-1217', '679568']"
    "test_subjects: ['sub-1047', 'sub-1050', 'sub-1080', '622236']"
    "test_subjects: ['sub-1146', 'sub-1135', 'sub-1156', '673455']"
    "test_subjects: ['sub-1164', 'sub-1235', 'sub-1281', '802844']"
    "test_subjects: ['sub-1245', 'sub-1061', 'sub-1115', '786569']"
    "test_subjects: ['sub-1209', 'sub-1191', 'sub-1222', '761957']"  
    "test_subjects: ['sub-1263', 'sub-1049', 'sub-1066', '715647']"  
    "test_subjects: ['sub-1024', 'sub-1178', 'sub-1019', '930449']"  
    "test_subjects: ['sub-1124', 'sub-1149', '889579']"
    "test_subjects: ['sub-1232', 'sub-1006', '599469']"
)

base_model_command="CUDA_VISIBLE_DEVICES=0 python main.py --num_workers 40 --config $config --test --exp_name $exp_name --ckpts $ckpts --exp_subset $exp_subset"
dice_score_command="python score_calculation/dice_score_universal.py --model $model --dataset_name $dataset_name --exp_name $exp_name --exp_subset $exp_subset --mapping_path $mapping_path"
# Loop through each subject_ids block
for i in "${!subject_blocks[@]}"
do

    echo "Adding subject_ids block $((i + 1)) to the config file..."
    echo "${subject_blocks[$i]}" >> "$dataset_config_file"
    echo "Running iteration $((i + 1)) with subject_ids block:"
    echo "${subject_blocks[$i]}"

    conda deactivate
    conda activate TractoBERT
    eval $base_model_command
    wait
    conda deactivate
    conda activate scilpy_env
    eval $dice_score_command
    wait


    echo "Removing subject_ids block $((i + 1)) from the config file..."
    sed -i "/^subject_ids: \[.*\]/d" "$dataset_config_file"
    echo "Iteration $((i + 1)) completed."
done