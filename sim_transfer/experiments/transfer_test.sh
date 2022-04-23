N=$1
pre=$2
post=$3
trans_hid=$4

./transfer_single.sh short_joints $N $pre $post $trans_hid
./transfer_single.sh long_joints $N $pre $post $trans_hid
./transfer_single.sh low_friction $N $pre $post $trans_hid