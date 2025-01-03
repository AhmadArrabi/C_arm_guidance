export LOG_DIR="root/logs/self_supervised" 
export ANNOTATIONS_DIR="root/annotations"

python ./src/train_self_supervised.py \
   --batch_size 32 \
   --epochs 20 \
   --log_dir=$LOG_DIR \
   --annotations_dir=$ANNOTATIONS_DIR \
   --exp_name experiment_1 \
   --model_name resnet34 
   