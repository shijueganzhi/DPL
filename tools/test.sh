CONFIG_FILE=configs/dpl/cityscapes_dpl_dual.py
CHECKPOINT_FILE=work_dirs/cityscapes_dpl_dual/latest.pth
OUTPUT_DIR=work_dirs/cityscapes_dpl_dual/vis

# test
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU --show-dir ${OUTPUT_DIR}