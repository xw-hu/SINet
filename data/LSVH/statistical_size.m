% use to determine the parameters in 'ROISplit' layer
% split_area is better to be the median area. 
% fluctuation_range can be determined by the graph.
% "a" is a parameter, test by validation set.

% Copyright (c) The Chinese University of Hong Kong
% HU Xiaowei, 2016
% Thanks for Zhaowei Cai [The Regents of the University of California]
% providing original code

clear all; close all;

%parameter:  
a=1/20;

% set root_dir to your KITTI directory
root_dir = '/home/xwhu/LSVH/';
image_dir = [root_dir 'images/'];
label_dir = [root_dir 'labels/'];

% choose which data list to generate
 dataType = 'trainval';
%dataType = 'test';
 
image_name = importdata(['ImageSets/' dataType '.txt']);

total_car_width = [];
total_car_height = [];

for i = 1:length(image_name)
  if (mod(i,500) == 0), fprintf('image idx: %i/%i\n', i, length(image_name)); end
   
  img_path = [image_dir image_name{i} '.jpg'];
  I=imread(img_path); 

  [imgH imgW channels]=size(I);
  
  % choose the right input size
   resize_w = 1344; resize_h = 768; Min_Height = 15;

  
  ratio_w = resize_w/imgW;
  ratio_h = resize_h/imgH; 
  
  objects = readLabel(label_dir, image_name{i});
  
  labels = []; labelidx = []; dontcareidx = [];
  for j = 1:numel(objects)
    obj = objects(j);
    if (obj.x2<=obj.x1 || obj.y2<=obj.y1)
      continue;
    end
    if (strcmp(obj.type,'1')) %car
      labels = cat(1,labels,1); labelidx = cat(1,labelidx,j); 
    elseif (strcmp(obj.type,'2')) %bus
      labels = cat(1,labels,2); labelidx = cat(1,labelidx,j); 
    elseif (strcmp(obj.type,'3')) %van
      labels = cat(1,labels,3); labelidx = cat(1,labelidx,j); 
    elseif (strcmp(obj.type,'4')) %DontCare
      dontcareidx = cat(1,dontcareidx,j);
    end
  end
    
  num_car = 0;
  
  num_objs = length(labelidx);

  for j = 1:num_objs
    idx = labelidx(j); object = objects(idx);
    ignore = 0;
    x1 = object.x1 * ratio_w; y1 = object.y1 * ratio_h;
    x2 = object.x2 * ratio_w; y2 = object.y2 * ratio_h;
    
    if labels(j)~=4 && ignore~=1  && (round(y2)-round(y1)+1) >= Min_Height   %car
        num_car = num_car+1;
    end
   
  end
  
  if num_car==0
      continue;
  end
  
  car_width = zeros(num_car,1);
  car_height = zeros(num_car,1);
  
  count_cat = 0;
  for j = 1:num_objs
    idx = labelidx(j); object = objects(idx);
    ignore = 0;
    x1 = object.x1 * ratio_w; y1 = object.y1 * ratio_h;
    x2 = object.x2 * ratio_w; y2 = object.y2 * ratio_h;
    
    if labels(j)~=4 && ignore~=1  && (round(y2) - round(y1)+1) >= Min_Height   %car
        count_cat = count_cat+1;
        car_width(count_cat,1) = round(x2) - round(x1);
        car_height(count_cat,1) = round(y2) - round(y1);
    end
  end
  
  total_car_width = cat(1,total_car_width, car_width);
  total_car_height = cat(1,total_car_height, car_height);
  
end


area = total_car_width.*total_car_height;
plot(total_car_width,total_car_height,'b.');
xlabel('car width');
ylabel('car height')
grid on;
hold on;


areaid = sort(area);

wid = sort(total_car_width);
hid = sort(total_car_height);

total_car = length(wid);

fprintf('The median of width %d\n',wid(ceil(total_car/2)));
fprintf('The median of height %d\n',hid(ceil(total_car/2)));

fprintf('The average of width %d\n',mean(wid));
fprintf('The average of height %d\n',mean(hid));

fprintf('The median of area %d\n',areaid(ceil(total_car/2)));

fprintf('The average of area %d\n',mean(areaid));

fr1=a*(mean(areaid)-areaid(ceil(total_car/2)));
fprintf('%f of the difference between median and mean: %d\n',a,fr1);


for i=1:total_car
    if total_car_width(i)*total_car_height(i)< areaid(ceil(total_car/2))
        plot(total_car_width(i),total_car_height(i),'r+');
    else
        plot(total_car_width(i),total_car_height(i),'b*');
    end
end

figure(1);
plot(wid(ceil(total_car/2)),hid(ceil(total_car/2)),'g+');
plot(mean(wid), mean(hid) ,'y*');

%%%calculate the variance of data
m = areaid(ceil(total_car/2));
var1 = 0;
var2 = 0;
for i=1:ceil(total_car/2)
    var1 = var1 + abs(areaid(i)-m);
end
std1 = var1/ceil(total_car/2);

for i=ceil(total_car/2)+1:total_car
    var2 = var2 + abs(areaid(i)-m);
end
std2 = var2/(total_car-ceil(total_car/2));

ratio = std2/std1;
fprintf('split_area %d\n',areaid(ceil(total_car/2)));
fprintf('fluctuation_range [both]: %f\n',fr1);

print(1,'-dpng','data_distribution'); 
