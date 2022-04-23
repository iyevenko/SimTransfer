N=$1
pre=$2
post=$3
trans_hid=$4
cd ..
python train_ppo.py --epochs $N --xml low_friction/walker2d_friction_80.xml --ac walker2d_base --pre $pre --post $post --trans_hid $trans_hid --exp_name transfer_friction_80
python train_ppo.py --epochs $N --xml low_friction/walker2d_friction_60.xml --ac walker2d_base --pre $pre --post $post --trans_hid $trans_hid --exp_name transfer_friction_60
python train_ppo.py --epochs $N --xml low_friction/walker2d_friction_40.xml --ac walker2d_base --pre $pre --post $post --trans_hid $trans_hid --exp_name transfer_friction_40
python train_ppo.py --epochs $N --xml low_friction/walker2d_friction_20.xml --ac walker2d_base --pre $pre --post $post --trans_hid $trans_hid --exp_name transfer_friction_20