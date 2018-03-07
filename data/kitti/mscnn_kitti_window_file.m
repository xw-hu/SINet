% Copyright (c) 2016 The Regents of the University of California
% see mscnn/LICENSE for details
% Written by Zhaowei Cai [zwcai-at-ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!

clear all; close all;

% set root_dir to your KITTI directory
root_dir = '/your/KITTI/path/';
addpath([root_dir 'devkit_object/devkit/matlab']);
image_dir = [root_dir 'training/image_2/'];
label_dir = [root_dir 'data_object_label_2/training/label_2/'];

% choose which data list to generate
dataType = 'train';
% dataType = 'val';
% dataType = 'trainval';

id_list = load(['ImageSets/' dataType '.txt']);
id_list = id_list+1;

file_name = sprintf('window_files/mscnn_window_file_kitti_vehicle_%s.txt',dataType);
fid = fopen(file_name, 'wt');

show = 1;
if (show)
  fig = figure(1); set(fig,'Position',[-30 30 960 300]);
  hd.axes = axes('position',[0.1,0.1,0.8,0.8]);
end

for i = 1:length(id_list)
  if (mod(i,500) == 0), fprintf('image idx: %i/%i\n', i, length(id_list)); end
  imgidx = id_list(i);
  img_path = sprintf([image_dir '%06i.png'],imgidx-1);
  I=imread(img_path); 
  if (show)
    imshow(I); axis(hd.axes,'image','off'); hold(hd.axes, 'on');
  end
  [imgH imgW channels]=size(I);
  
  objects = readLabels(label_dir, imgidx-1);
  
  fprintf(fid, '# %d\n', i-1);
  fprintf(fid, '%s\n', img_path);
  fprintf(fid, '%d\n%d\n%d\n', channels, imgH, imgW);
  
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
    
  num_objs = length(labelidx);
  fprintf(fid, '%d\n', num_objs);
  for j = 1:num_objs
    idx = labelidx(j); object = objects(idx);
    ignore = 0;
    x1 = object.x1; y1 = object.y1;
    x2 = object.x2; y2 = object.y2;
    w = x2-x1+1; h = y2-y1+1;
    trunc = object.truncation;  occ = object.occlusion; 
    % ignore largely occluded and truncated objects
    if (occ>=2 || trunc>=0.5) 
      ignore = 1;
    end
    fprintf(fid, '%d %d %d %d %d %d\n', labels(j), ignore, round(x1), round(y1), round(x2), round(y2));
    
    if (show)
      if (ignore), color = 'g'; else color = 'r'; end
      rectangle('Position', [x1 y1 w h],'LineWidth',2,'edgecolor',color);   
      text(x1+0.5*w,y1,num2str(labels(j)),'color','r','BackgroundColor','k','HorizontalAlignment',...
         'center','VerticalAlignment','bottom','FontWeight','bold','FontSize',8);
    end
  end
  
  num_dontcare = length(dontcareidx);
  fprintf(fid, '%d\n', num_dontcare);
  for j  = 1:num_dontcare
    idx = dontcareidx(j); object = objects(idx);
    x1 = object.x1; y1 = object.y1;
    x2 = object.x2; y2 = object.y2;
    fprintf(fid, '%d %d %d %d\n', round(x1), round(y1), round(x2), round(y2));
    if (show)
      rectangle('Position', [x1 y1 x2-x1 y2-y1],'LineWidth',2.5,'edgecolor','y');
    end
  end
  if (show), pause(0.01); end
end

fclose(fid);

