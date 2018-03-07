% help to determine the anchor size
% HU Xiaowei, The Chinese University of Hong Kong 

clear all; close all;

% set root_dir to your KITTI directory
root_dir = '/your/KITTI/path/';
addpath([root_dir 'devkit_object/matlab']);
image_dir = [root_dir 'data_object_image_2/training/image_2/'];
label_dir = [root_dir 'data_object_image_2/training/label_2/'];

% choose which data list to generate
 dataType = 'train';
% dataType = 'val';
% dataType = 'trainval';

id_list = load(['ImageSets/' dataType '.txt']);
id_list = id_list+1;

%total_bbs = [];
total_car_width = [];
total_car_height = [];

for i = 1:length(id_list)
  if (mod(i,500) == 0), fprintf('image idx: %i/%i\n', i, length(id_list)); end
  imgidx = id_list(i);
  img_path = sprintf([image_dir '%06i.png'],imgidx-1);
  I=imread(img_path); 

  [imgH imgW channels]=size(I);
  
  % choose the right input size
  resize_w = 1920; resize_h = 576; Min_Height = 35;
  % resize_w = 2560; resize_h = 768; Min_Height = 45; 
  % resize_w = 1280; resize_h = 384; Min_Height = 25; 
  % resize_w = 864; resize_h = 256; Min_Height = 16;
  
  ratio_w = resize_w/imgW;
  ratio_h = resize_h/imgH; 
  
  objects = readLabels(label_dir, imgidx-1);
  
  labels = []; labelidx = []; dontcareidx = [];
  for j = 1:numel(objects)
    obj = objects(j);
    if (obj.x2<=obj.x1 || obj.y2<=obj.y1)
      continue;
    end
    if (strcmp(obj.type,'Car'))
      labels = cat(1,labels,1); labelidx = cat(1,labelidx,j); 
    elseif (strcmp(obj.type,'Van'))
      labels = cat(1,labels,2); labelidx = cat(1,labelidx,j); 
    elseif (strcmp(obj.type,'Truck'))
      labels = cat(1,labels,3); labelidx = cat(1,labelidx,j); 
    elseif (strcmp(obj.type,'Tram') || strcmp(obj.type,'Misc'))
      labels = cat(1,labels,4); labelidx = cat(1,labelidx,j); 
    elseif (strcmp(obj.type,'DontCare'))
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
    w = x2-x1+1; h = y2-y1+1;
    trunc = object.truncation;  occ = object.occlusion; 
    % ignore largely occluded and truncated objects
    if (occ>=2 || trunc>=0.5) 
      ignore = 1;
    end
    
    if labels(j)==1 && ignore~=1  && (round(y2)-round(y1)+1) >= Min_Height   %car (min_height)
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
    w = x2-x1+1; h = y2-y1+1;
    trunc = object.truncation;  occ = object.occlusion; 
    % ignore largely occluded and truncated objects
    if (occ>=2 || trunc>=0.5) 
      ignore = 1;
    end
    
    if labels(j)==1 && ignore~=1  && (round(y2) - round(y1)+1) >= Min_Height   %car
        count_cat = count_cat+1;
%         bbs(count_cat,1)=x1;
%         bbs(count_cat,2)=y1;
%         bbs(count_cat,3)=w;
%         bbs(count_cat,4)=h;
        car_width(count_cat,1) = round(x2) - round(x1);
        car_height(count_cat,1) = round(y2) - round(y1);
    end
    
    %fprintf(fid, '%d %d %d %d %d %d\n', labels(j), ignore, round(x1), round(y1), round(x2), round(y2));
  end
  
  %total_bbs = cat(1,total_bbs, bbs);
  total_car_width = cat(1,total_car_width, car_width);
  total_car_height = cat(1,total_car_height, car_height);

end

area = total_car_width.*total_car_height;
car_num = length(area);


sort_width = sort(total_car_width);
sort_height = sort(total_car_height);

%%%K-means
K=7; %7 outputs for proposals

ini_width = [sort_width(ceil(car_num/8));sort_width(ceil(car_num/20*7));
    sort_width(ceil(car_num/8*3)); sort_width(ceil(car_num/5*3)); 
    sort_width(ceil(car_num/8*5));sort_width(ceil(car_num/20*17));
    sort_width(ceil(car_num/8*7))]; 

%[Idx,field_w,sumD,D]=kmeans(total_car_width,K,'Start',[60;84;120;168;240;336;480]);

[Idx,field_w,sumD,D]=kmeans(total_car_width,K,'Start',ini_width);


for i=1:K
    fprintf('field_w %d: %f\n',i,field_w(i));
end

ini_height = [sort_height(ceil(car_num/8));sort_height(ceil(car_num/20*7));
    sort_height(ceil(car_num/8*3)); sort_height(ceil(car_num/5*3)); 
    sort_height(ceil(car_num/8*5));sort_height(ceil(car_num/20*17));
    sort_height(ceil(car_num/8*7))]; 

[Idx,field_h,sumD,D]=kmeans(total_car_height,K,'Start',ini_height);

for i=1:K
    fprintf('field_h %d: %f\n',i,field_h(i));
end
