export LOG_DIR="root/logs/self_supervised" 
export DATA_DIR="root/data"
export ANNOTATIONS_DIR="root/annotations"

python ./src/train_self_supervised.py \
   --batch_size 32 \
   --epochs 20 \
   --log_dir=$LOG_DIR \
   --data_dir=$DATA_DIR \
   --annotations_dir=$ANNOTATIONS_DIR \
   --exp_name experiment_1 \
   --model_name resnet34 
   