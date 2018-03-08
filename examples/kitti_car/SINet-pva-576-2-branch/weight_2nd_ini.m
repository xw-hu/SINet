% Copyright (c) 2016 The Chinese University of Hong Kong
% Written by Xiaowei Hu [xwhu@cse.cuhk.edu.hk]

addpath('../../../matlab');
caffe.reset_all();

%caffe.set_mode_cpu();
caffe.set_mode_gpu();
caffe.set_device(0);
model1='trainval_2nd_ini.prototxt';
model2='trainval_2nd.prototxt';

weight1='../../../models/PVA/original.model';
weight2='SINet_kitti_train_1st_iter_10000.caffemodel';

net1=caffe.Net(model1,weight1,'train');
net2=caffe.Net(model2,weight2,'train');

fc6_1=net1.layers('fc6').params(1).get_data();
fc6_2=net1.layers('fc6').params(2).get_data();


net2.layers('fc6_small').params(1).set_data(fc6_1);
net2.layers('fc6_small').params(2).set_data(fc6_2);

net2.layers('fc6/bn_small').params(1).set_data(net1.layers('fc6/bn').params(1).get_data()); 
net2.layers('fc6/bn_small').params(2).set_data(net1.layers('fc6/bn').params(2).get_data()); 
net2.layers('fc6/bn_small').params(3).set_data(net1.layers('fc6/bn').params(3).get_data()); 
net2.layers('fc6/scale_small').params(1).set_data(net1.layers('fc6/scale').params(1).get_data()); 
net2.layers('fc6/scale_small').params(2).set_data(net1.layers('fc6/scale').params(2).get_data()); 

net2.layers('fc6_large').params(1).set_data(fc6_1);
net2.layers('fc6_large').params(2).set_data(fc6_2);

net2.layers('fc6/bn_large').params(1).set_data(net1.layers('fc6/bn').params(1).get_data()); 
net2.layers('fc6/bn_large').params(2).set_data(net1.layers('fc6/bn').params(2).get_data()); 
net2.layers('fc6/bn_large').params(3).set_data(net1.layers('fc6/bn').params(3).get_data()); 
net2.layers('fc6/scale_large').params(1).set_data(net1.layers('fc6/scale').params(1).get_data()); 
net2.layers('fc6/scale_large').params(2).set_data(net1.layers('fc6/scale').params(2).get_data()); 


net2.save('SINet_kitti_train_2nd_iter_initial.caffemodel');

caffe.reset_all();
