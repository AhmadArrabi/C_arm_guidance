export LOG_DIR="root/logs/classifier" 
export CHECKPOINT_SELF_SUPERVISED="root/logs/self_supervised/experiment_1/checkpoints/chkpt_e100.pt"
export ANNOTATIONS_DIR="root/annotations"

python ./src/train_classifier.py \
   --batch_size 64 \
   --epochs 20 \
   --log_dir=$LOG_DIR \
   --annotations_dir=$ANNOTATIONS_DIR \
   --exp_name experiment_1 \
   --model_name resnet34 \
   --pretrained_weights position \
   --checkpoint_self_supervised=$CHECKPOINT_SELF_SUPERVISED \
   --classifier_layers 1 
   
