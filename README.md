## Dependencies
Python 3.6
PyTorch >= 1.0.0
numpy
skimage
imageio
matplotlib
tqdm
cv2 >= 3.xx (Only if you want to use video input/output)
Matlab R2014a

## Code
Clone this repository into any place you want.

git clone https://github.com/qiyu-lin/MRCAN_MISSA.git
cd MRCAN_MISSA

## Dataset
Our Wafer datasets now can be accessed at {https://pan.baidu.com/s/1AGrJTjLuw0HhQojnBxvcPg},extraction code:trbs

## Multi-scale residual channel attention network
Train
python main.py --model MRCAN --save MRCAN_BIX4_wafer --scale 4 --n_resgroups 8 --n_resblocks 20 --n_feats 64 --reset --chop --save_results --print_model --patch_size 192

Test
python main.py --model MRCAN --n_resgroups 8 --n_resblocks 20 --n_feats 64 --pre_train /home/user1/MRCAN-PyTorch-master/experiment/MRCAN_BIX4_wafer/model/model_best.pt --test_only --save_results --chop --save MRCAN_BIX4_result

## Multi-Strategy Improved Sparrow Search Algorithm

Run main_proposed.m in the folder MISSA.