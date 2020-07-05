pip install -r requirement.txt
feat=$1
net=$2
# opt=$3
# bs=$4
# ol=$3
log_dir="./log/$feat""_""$net"
touch $log_dir
echo $log_dir
python config.py
CUDA_VISIBLE_DEVICES=7 python __main__.py $feat $ol > $log_dir
