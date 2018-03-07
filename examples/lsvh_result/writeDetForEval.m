% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!
function writeDetForEval
% clear and close everything
clear all; close all;
root_dir = '/home/xwhu/LSVH/';
image_dir = [root_dir 'images/'];
label_dir = [root_dir 'labels/'];

data_set = 'test';
is_gt_available = 1;
list_dir = ['../../data/LSVH/ImageSets/' data_set '.txt'];
image_name = importdata(list_dir);
nimages = length(image_name);

% load detection results
car_dets_path = '../lsvh_vehicle/detections/SINet_LSVH_result_car.txt';
if (exist(car_dets_path))
  car_dets = load(car_dets_path);
else
  car_dets = zeros(0,6);
end

bus_dets_path = '../lsvh_vehicle/detections/SINet_LSVH_result_bus.txt';
if (exist(bus_dets_path))
  bus_dets = load(bus_dets_path);
else
  bus_dets = zeros(0,6);
end

van_dets_path = '../lsvh_vehicle/detections/SINet_LSVH_result_van.txt';
if (exist(van_dets_path))
  van_dets = load(van_dets_path);
else
  van_dets = zeros(0,6);
end

score_scale = 1000;
comp_id = 'lsvh_detection';
result_dir = [data_set '/' comp_id '/'];
save_dir = [result_dir 'data/'];
if (~exist(save_dir)), mkdir(save_dir); end

for i = 1:nimages
    if (mod(i,1000)==0), fprintf('idx: %i / %i\n',i,nimages); end
    objects=[]; num = 0;
    ss = strsplit(image_name{i},'-');
    scene = ss{1};
    % car
    idx_car = find(car_dets(:,1)==i);
    bbs_car = car_dets(idx_car,2:6);
    bbs_car(:,3:4) = bbs_car(:,1:2)+bbs_car(:,3:4);
    for j = 1:size(bbs_car,1)
      num = num+1;
      objects(num).type = '1'; objects(num).score = bbs_car(j,5)*score_scale;
      objects(num).x1 = bbs_car(j,1); objects(num).y1 = bbs_car(j,2);
      objects(num).x2 = bbs_car(j,3); objects(num).y2 = bbs_car(j,4);
    end
    
    % bus
    idx_bus = find(bus_dets(:,1)==i);
    bbs_bus = bus_dets(idx_bus,2:6);
    bbs_bus(:,3:4) = bbs_bus(:,1:2)+bbs_bus(:,3:4);
    for j = 1:size(bbs_bus,1)
      num = num+1;
      objects(num).type = '2'; objects(num).score = bbs_bus(j,5)*score_scale;
      objects(num).x1 = bbs_bus(j,1); objects(num).y1 = bbs_bus(j,2);
      objects(num).x2 = bbs_bus(j,3); objects(num).y2 = bbs_bus(j,4);
    end
    
    % van
    idx_van = find(van_dets(:,1)==i);
    bbs_van = van_dets(idx_van,2:6);
    bbs_van(:,3:4) = bbs_van(:,1:2)+bbs_van(:,3:4);
    for j = 1:size(bbs_van,1)
      num = num+1;
      objects(num).type = '3'; objects(num).score = bbs_van(j,5)*score_scale;
      objects(num).x1 = bbs_van(j,1); objects(num).y1 = bbs_van(j,2);
      objects(num).x2 = bbs_van(j,3); objects(num).y2 = bbs_van(j,4);
    end
    
    img_idx = image_name{i};
    writeLabels(objects,save_dir,img_idx);
end


if (is_gt_available)
  plot_dir = [result_dir 'plot/'];
  %if (~exist(plot_dir)), mkdir(save_dir); end
  % input arguments [gt_dir, result_dir, list_dir];
  command_line = sprintf('eval/evaluate_object %s %s %s', label_dir,result_dir,list_dir);
  system(command_line);
  plot_set = dir([plot_dir '*.txt']);
  for i = 1:length(plot_set)
    results = importdata([plot_dir plot_set(i).name]);
    x = results(:,1); fig=figure(i);
    h1 = plot(x,results(:,2),'LineWidth',3,'Color','r'); hold on;
    sparse_legend = sprintf('%s %%%.02f','Sparse',100*xVOCap(results(:,1),results(:,2)));
    h2 = plot(x,results(:,5),'LineWidth',3,'Color',[0,0,0]); hold on;
    crowded_legend = sprintf('%s %%%.02f','Crowded',100*xVOCap(results(:,1),results(:,5)));
    hd=legend([h1 h2], sparse_legend, crowded_legend);
    set(hd,'FontSize',18,'Location','SouthWest'); grid;
    tt=get(gca,'Title'); title(plot_set(i).name(1:end-14)); set(tt,'FontSize',18); 
    saveas(fig,[result_dir plot_set(i).name(1:end-4) '.png'])
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% voc2011
function ap = xVOCap(rec,prec)
% From the PASCAL VOC 2011 devkit

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end

