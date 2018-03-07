function bbs = bbAvgNms( bbs, varargin )
%EXAMPLE
% pAvg.type = 'maxg'; pAvg.overlap = 0.5;
% pAvg.ovrDnm = 'union'; pAvg.merge_overlap = 0.8;
% bbset=bbAvgNms(bbset,pAvg);

% bbset:  x1,y1,width,heihgt,confidence,num
% [599.632385253906,159.617630004883,32.3860015869141,30.6816520690918,0.540845096111298,1]
% [597.602050781250,157.192382812500,35.0840644836426,31.9155082702637,0.184505060315132,2]

% Bounding box (bb) avgerage (nms).
%
% type=='max': nms of bbs using area of overlap criteria. For each pair of
% bbs, if their overlap, defined by:
%  overlap(bb1,bb2) = area(intersect(bb1,bb2))/area(union(bb1,bb2))
% is greater than overlap, then the bb with the lower score is suppressed.
% In the Pascal critieria two bbs are considered a match if overlap>=.5. If
% ovrDnm='min', the 'union' in the above formula is replaced with 'min'.
%
% type=='maxg': Similar to 'max', except performs the nms in a greedy
% fashion. Bbs are processed in order of decreasing score, and, unlike in
% 'max' nms, once a bb is suppressed it can no longer suppress other bbs.

% USAGE
%  bbs = bbNms( bbs, [varargin] )
%
% INPUTS
%  bbs        - original bbs (must be of form [x y w h wt bbType])
%  varargin   - additional params (struct or name/value pairs)
%   .type       - ['max'] 'max', 'maxg', 'ms', 'cover', or 'none'
%   .thr        - [-inf] threshold below which to discard (0 for 'ms')
%   .maxn       - [inf] if n>maxn split and run recursively (see above)
%   .radii      - [.15 .15 1 1] supression radii ('ms' only, see above)
%   .overlap    - [.5] area of overlap for bbs
%   .ovrDnm     - ['union'] area of overlap denominator ('union' or 'min')
%   .resize     - {} parameters for bbApply('resize')
%   .separate   - [0] run nms separately on each bb type (bbType)
%   .merge_overlap   -[0.9] area of overlap for average 
%
% Piotr's Image&Video Toolbox      Version 2.60
% Copyright 2012 Piotr Dollar.  [pdollar-at-caltech.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see external/bsd.txt]
% HU Xiaowei modifies for averaging the coordinate when IoU is larger than the second threshold
% 2016 The Chinese University of Hong Kong

% get parameters
dfs={'type','max','thr',[],'maxn',inf,'radii',[.15 .15 1 1],...
  'overlap',.5,'ovrDnm','union','resize',{},'separate',0,'merge_overlap',.9;};
[type,thr,maxn,radii,overlap,ovrDnm,resize,separate,merge_overlap] = ...
  getPrmDflt(varargin,dfs,1);
if(isempty(thr)), if(strcmp(type,'ms')), thr=0; else thr=-inf; end; end
if(strcmp(ovrDnm,'union')), ovrDnm=1; elseif(strcmp(ovrDnm,'min')),
  ovrDnm=0; else assert(false); end
assert(maxn>=2); assert(numel(overlap)==1); assert(numel(merge_overlap)==1);

% discard bbs below threshold and run nms1
if(isempty(bbs)), bbs=zeros(0,5); end; if(strcmp(type,'none')), return; end
kp=bbs(:,5)>thr; bbs=bbs(kp,:); if(isempty(bbs)), return; end
if(~isempty(resize)), bbs=bbApply('resize',bbs,resize{:}); end
pNms1={type,thr,maxn,radii,overlap,0, merge_overlap};
if(~separate || size(bbs,2)<6), bbs=nms1(bbs,pNms1{:}); else
  ts=unique(bbs(:,6)); m=length(ts); bbs1=cell(1,m);
  for t=1:m, bbs1{t}=nms1(bbs(bbs(:,6)==ts(t),:),pNms1{:}); end
  bbs=cat(1,bbs1{:});
end

  function bbs = nms1( bbs, type, thr, maxn, radii, overlap, isy, merge_overlap)
    % if big split in two, recurse, merge, then run on merged
    if( size(bbs,1)>maxn )
      n2=floor(size(bbs,1)/2); [~,ord]=sort(bbs(:,1+isy)+bbs(:,3+isy)/2);
      bbs0=nms1(bbs(ord(1:n2),:),type,thr,maxn,radii,overlap,~isy);
      bbs1=nms1(bbs(ord(n2+1:end),:),type,thr,maxn,radii,overlap,~isy);
      bbs=[bbs0; bbs1];
    end
    % run actual nms on given bbs
    switch type
      case 'max', bbs = nmsMax(bbs,overlap,0,ovrDnm, merge_overlap);
      case 'maxg', bbs = nmsMax(bbs,overlap,1,ovrDnm, merge_overlap);
      otherwise, error('unknown type: %s',type);
    end
  end

  function bbs = nmsMax( bbs, overlap, greedy, ovrDnm, merge_overlap )
    % for each i suppress all j st j>i and area-overlap>overlap
    [~,ord]=sort(bbs(:,5),'descend'); bbs=bbs(ord,:);
    n=size(bbs,1); kp=true(1,n); as=bbs(:,3).*bbs(:,4); %area
    
    count = ones(1,n); %mi = mi + (1/ni) * (x-mi) [ni]
    
    xs=bbs(:,1); xe=bbs(:,1)+bbs(:,3); ys=bbs(:,2); ye=bbs(:,2)+bbs(:,4); %xy l r
    for i=1:n, if(greedy && ~kp(i)), continue; end
      for j=(i+1):n, if(kp(j)==0), continue; end
        iw=min(xe(i),xe(j))-max(xs(i),xs(j)); if(iw<=0), continue; end
        ih=min(ye(i),ye(j))-max(ys(i),ys(j)); if(ih<=0), continue; end
        o=iw*ih; if(ovrDnm), u=as(i)+as(j)-o; else u=min(as(i),as(j)); end
        o=o/u; if(o>overlap), kp(j)=0; end
        
        %%average the coordinate  Hu Xiaowei 2016.12
        if (o>merge_overlap)
            count(i) = count(i)+1;
            %sequencial apdate
%             bbs(i,1) = bbs(i,1) + 1/count(i) * (bbs(j,1)-bbs(i,1)); %%mi = mi + (1/ni) * (x-mi)
%             bbs(i,2) = bbs(i,2) + 1/count(i) * (bbs(j,2)-bbs(i,2)); %%mi = mi + (1/ni) * (x-mi)
%             bbs(i,3) = bbs(i,3) + 1/count(i) * (bbs(j,3)-bbs(i,3)); %%mi = mi + (1/ni) * (x-mi)
%             bbs(i,4) = bbs(i,4) + 1/count(i) * (bbs(j,4)-bbs(i,4)); %%mi = mi + (1/ni) * (x-mi)
            bbs(i,1) = bbs(i,1) + bbs(j,1);
            bbs(i,2) = bbs(i,2) + bbs(j,2);
            bbs(i,3) = bbs(i,3) + bbs(j,3);
            bbs(i,4) = bbs(i,4) + bbs(j,4);
        end
      end
      
      if(count(i)~=1)
          bbs(i,1) = bbs(i,1) / count(i);
          bbs(i,2) = bbs(i,2) / count(i);
          bbs(i,3) = bbs(i,3) / count(i);
          bbs(i,4) = bbs(i,4) / count(i);
      end
    end
    bbs=bbs(kp>0,:);
  end
end
