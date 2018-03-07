% help to determine the anchor size
% HU Xiaowei, The Chinese University of Hong Kong 

clear all; close all;

% set root_dir to your LSVH directory
root_dir = '/home/xwhu/LSVH/';
image_dir = [root_dir 'images/'];
label_dir = [root_dir 'labels/'];

% choose which data list to generate
 dataType = 'trainval';

image_name = importdata(['ImageSets/' dataType '.txt']);

%total_bbs = [];
total_car_width = [];
total_car_height = [];

% r = zeros(length(image_name),1);
% g = zeros(length(image_name),1);
% b = zeros(length(image_name),1);

for i = 1:length(image_name)
  if (mod(i,500) == 0), fprintf('image idx: %i/%i\n', i, length(image_name)); end
  
  img_path = [image_dir image_name{i} '.jpg'];
  I=imread(img_path); 

  [imgH imgW channels]=size(I);
  
  %r(i) = mean(mean(I(:,:,1)));
  %g(i) = mean(mean(I(:,:,2)));
  %b(i) = mean(mean(I(:,:,3)));
  
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
    
    if labels(j)~=4 && ignore~=1  && (round(y2)-round(y1)+1) >= Min_Height   %car (min_height)
        num_car = num_car+1;
    end
   
  end
  
  if num_car==0
      continue;
  end
  
  %bbs = zeros(num_car,4); %x1,y2,w,h;
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
car_num = length(area);


sort_width = sort(total_car_width);
sort_height = sort(total_car_height);

%%%K-means
K=8; %8 outputs for proposals

ini_width = [sort_width(ceil(car_num/8));sort_width(ceil(car_num/20*6));
    sort_width(ceil(car_num/8*3)); sort_width(ceil(car_num/20*10)); 
    sort_width(ceil(car_num/8*5));sort_width(ceil(car_num/20*14));
    sort_width(ceil(car_num/8*7));sort_width(ceil(car_num/20*19))]; 

[Idx,field_w,sumD,D]=kmeans(total_car_width,K,'Start',ini_width);


for i=1:K
    fprintf('field_w %d: %f\n',i,field_w(i));
end

ini_height = [sort_height(ceil(car_num/8));sort_height(ceil(car_num/20*6));
    sort_height(ceil(car_num/8*3)); sort_height(ceil(car_num/20*10)); 
    sort_height(ceil(car_num/8*5));sort_height(ceil(car_num/20*14));
    sort_height(ceil(car_num/8*7));sort_width(ceil(car_num/20*19))]; 

[Idx,field_h,sumD,D]=kmeans(total_car_height,K,'Start',ini_height);

for i=1:K
    fprintf('field_h %d: %f\n',i,field_h(i));
end

% fprintf('mean_b: %f\n',mean(b(:)));
% fprintf('mean_g: %f\n',mean(g(:)));
% fprintf('mean_r: %f\n',mean(r(:)));