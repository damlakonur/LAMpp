# step1. set TRAIN_CONFIG path to config file

TRAIN_CONFIG="configs/inference/inference_lam_cafca.yaml"
MODEL_NAME="exps/releases/lam/lam-20k/step_045500/"
IMAGE_INPUT="sh ./scripts/install/install_cu118.sh"
MOTION_SEQS_DIR="assets/sample_motion/export/The_Shawshank_Redemption/"
CAFCA_SUBJECT_ID_SINGLE="32" # Specify the subject ID for single inference
CAFCA_CAMERA_ID_SINGLE="C13"  # Specify the camera ID for single inference
CAFCA_DRIVING_CAMERA_ID_SINGLE="C13"  # Specify the camera ID for single inference


TRAIN_CONFIG=${1:-$TRAIN_CONFIG}
MODEL_NAME=${2:-$MODEL_NAME}
IMAGE_INPUT=${3:-$IMAGE_INPUT}
MOTION_SEQS_DIR=${4:-$MOTION_SEQS_DIR}

echo "TRAIN_CONFIG: $TRAIN_CONFIG"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "MODEL_NAME: $MODEL_NAME"
echo "CAFCA_SUBJECT_ID_SINGLE (if use_cafca_dataset=True): $CAFCA_SUBJECT_ID_SINGLE"
echo "CAFCA_CAMERA_ID_SINGLE (if use_cafca_dataset=True): $CAFCA_CAMERA_ID_SINGLE"
echo "CAFCA_DRIVING_CAMERA_ID_SINGLE (if use_cafca_dataset=True): $CAFCA_DRIVING_CAMERA_ID_SINGLE"
echo "MOTION_SEQS_DIR: $MOTION_SEQS_DIR"


MOTION_IMG_DIR=null
SAVE_PLY=true
SAVE_IMG=true
VIS_MOTION=false
MOTION_IMG_NEED_MASK=true
RENDER_FPS=30
MOTION_VIDEO_READ_FPS=30
EXPORT_VIDEO=false
CROSS_ID=true
TEST_SAMPLE=false
USE_CAFCA_DATASET=true
GAGA_TRACK_TYPE=""

device=0
nodes=0

export PYTHONPATH=$PYTHONPATH:$pwd


CUDA_VISIBLE_DEVICES=$device python -m lam.launch infer.infer --config $TRAIN_CONFIG \
        model_name=$MODEL_NAME image_input=$IMAGE_INPUT \
        use_cafca_dataset=$USE_CAFCA_DATASET \
        cafca_subject_id_for_single_infer=$CAFCA_SUBJECT_ID_SINGLE \
        cafca_camera_id_for_single_infer=$CAFCA_CAMERA_ID_SINGLE \
        cafca_driving_camera_id_for_single=$CAFCA_DRIVING_CAMERA_ID_SINGLE \
        export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS \
        save_ply=$SAVE_PLY save_img=$SAVE_IMG \
        cross_id=$CROSS_ID \
        rank=$device nodes=$nodes

        
