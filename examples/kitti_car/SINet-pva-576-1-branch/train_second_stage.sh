GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver_2nd.prototxt \
  --weights=SINet_kitti_train_2nd_iter_initial.caffemodel \
  --gpu=0  2>&1 | tee log_2nd.txt
