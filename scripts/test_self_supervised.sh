export CHECKPOINT_PATH="root/logs/self_supervised/experiment_1/checkpoints/chkpt_e100.pt"
export ANNOTATIONS_DIR="root/annotations"

python ./src/test_classifier.py \
   --batch_size 64 \
   --model_name resnet34 \
   --annotations_dir=$ANNOTATIONS_DIR \
   --checkpoint_path=$CHECKPOINT_PATH 
