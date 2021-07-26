# The writefile magic command can be used to create and save a file

MODEL=$1
DEVICE=$2
VIDEO=$3
QUEUE=$4
OUTPUT=$5
PEOPLE=$6

mkdir -p $5

if [ $DEVICE = "HETERO:FPGA,CPU" ]; then
    #Environment variables and compilation for edge compute nodes with FPGAs
    source /opt/intel/init_openvino.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg1_bitstreams/2019R4_PL1_FP16_MobileNet_Clamp.aocx
fi

python3 person_detect.py  --model ${MODEL} \
                                --visualise \
                                --queue_param ${QUEUE} \
                                --device ${DEVICE} \
                                --video ${VIDEO}\
                                --output_path ${OUTPUT}\
                                --max_people ${PEOPLE} \
