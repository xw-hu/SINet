clear;
clc;

files = dir('/home/xwhu/LSVH/labels/*.txt');
len = length(files);

fidtest = fopen('test.txt','w');
fidtrain = fopen('train.txt','w');

for i=1:len
    
    txt_name = files(i).name;
    
    if ~isempty(strfind(txt_name,'Sparse-08'))
        fprintf(fidtest,'%s\n', txt_name(1:end-4));
  
    else if ~isempty(strfind(txt_name,'Sparse-12'))
        fprintf(fidtest,'%s\n', txt_name(1:end-4));

        else if ~isempty(strfind(txt_name,'Sparse-19'))
                fprintf(fidtest,'%s\n', txt_name(1:end-4));
    
            else if ~isempty(strfind(txt_name,'Sparse-22'))
                    fprintf(fidtest,'%s\n', txt_name(1:end-4));
                else if  ~isempty(strfind(txt_name,'Sparse'))
                        fprintf(fidtrain,'%s\n', txt_name(1:end-4));
                    end
                end
            end
        end
        
    end
end