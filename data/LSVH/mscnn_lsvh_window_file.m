clear all; close all;

% set root_dir to your LSVH directory
root_dir = '/home/xwhu/LSVH/';
image_dir = [root_dir 'images/'];
label_dir = [root_dir 'labels/'];

% choose which data list to generate
 dataType = 'train';
% dataType = 'test';

image_name = importdata(['ImageSets/' dataType '.txt']);

file_name = sprintf('window_files/mscnn_window_file_lsvh_vehicle_%s.txt',dataType);
fid = fopen(file_name, 'wt');

show = 0;
if (show)
  fig = figure(1); set(fig,'Position',[-30 30 960 300]);
  hd.axes = axes('position',[0.1,0.1,0.8,0.8]);
end

for i = 1:length(image_name)
  if (mod(i,500) == 0), fprintf('image idx: %i/%i\n', i, length(image_name)); end
 
  img_path = [image_dir image_name{i} '.jpg'];
  I=imread(img_path); 
  if (show)
    imshow(I); axis(hd.axes,'image','off'); hold(hd.axes, 'on');
  end
  [imgH imgW channels]=size(I);
  
  objects = readLabel(label_dir, image_name{i});
  
  fprintf(fid, '# %d\n', i-1);
  fprintf(fid, '%s\n', img_path);
  fprintf(fid, '%d\n%d\n%d\n', channels, imgH, imgW);
  
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
    
  num_objs = length(labelidx);
  fprintf(fid, '%d\n', num_objs);
  for j = 1:num_objs
    idx = labelidx(j); object = objects(idx);
    ignore = 0;
    x1 = object.x1; y1 = object.y1;
    x2 = object.x2; y2 = object.y2;
    % ignore largely occluded and truncated objects
    fprintf(fid, '%d %d %d %d %d %d\n', labels(j), ignore, round(x1), round(y1), round(x2), round(y2));
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

