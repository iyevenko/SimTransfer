exp_name=$1
N=$2
pre=$3
post=$4
trans_hid=$5
cd ..
python train_ppo.py --epochs $N --xml walker2d_$exp_name.xml --ac walker2d_base --pre $pre --post $post --trans_hid $trans_hid --exp_name transfer_$exp_name