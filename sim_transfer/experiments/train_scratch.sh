N=$1
cd ..
# Base environment
python train_ppo.py --epochs $N --exp_name walker2d_base
cp /home/iyevenko/Documents/spinningup/data/walker2d_base/walker2d_base_s0/pyt_save/model.pt models/walker2d_base/model.pt
# Custom environments
python train_ppo.py --epochs $N --xml walker2d_low_friction.xml --exp_name walker2d_low_friction
python train_ppo.py --epochs $N --xml walker2d_short_joints.xml --exp_name walker2d_short_joints
python train_ppo.py --epochs $N --xml walker2d_long_joints.xml --exp_name walker2d_long_joints