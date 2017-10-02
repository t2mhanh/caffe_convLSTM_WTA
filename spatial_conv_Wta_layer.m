% function spatial_conv_Wta_layer()
clear all
rand('seed',1)
in = rand(4,4,3,2);
[~,~,c_,n_] = size(in);
n = c_*n_;
[W,H,C,~] = size(in);
out_ = WtaForward(n,in_);
% parfor idx = 1:n    
%     c = mod(idx-1,C)+1;
%     n = floor((idx-1)/C) + 1;
%     data_in_start = ((n-1)*C + c-1)*H*W;
% %     out(data_in_start:data_in_start + H*W) = 0;
%     maxidx = 0;
%     maxval = -1;
%     
%     for h = 1:H
%         for w = 1:W
%             index = (h-1)*W + w-1 + 1;
%             if (in(index + data_in_start) > maxval)
%                 maxval = in(index + data_in_start);
%                 maxidx = index + data_in_start;
%             end              
%         end
%     end
%     out_maxidx(idx) = maxidx;
%     out_maxval(idx) = maxval;    
% end
% delete(gcp)
% out =  zeros(size(in));
% for idx = 1:length(out_maxidx)
%     out(out_maxidx(idx)) = out_maxval(idx);
% end
end

function out = WtaForward(n,in)
% parpool()
parfor i = 1:n
%     W = size(in,1);
%     H = size(in,2);
%     C = size(in,3);
%     N = size(in,4);
    [W,H,C,N] = size(in);
    c = floor(((i-1)/W/H),C)+1;
    n = floor((i-1)/W/H/C) + 1;
    data_in_start = ((n-1)*C + c-1)*H*W+1;
%     out(data_in_start:data_in_start + H*W) = 0;
    maxidx = 0;
    maxval = -1;
    for w = 1:W
        for h = 1:H
            index = (h-1)*W + w-1 + 1;
            if in(index + data_in_start) > maxval;
                maxval = in(index + data_in_start);
                maxidx(i) = index + data_in_start;
            end              
        end
    end
    out_maxidx(i) = maxidx;
    out_maxval(i) = maxval;    
end
delete(gcp)
out =  zeros(size(in));
for idx = 1:length(out_maxidx)
    out(out_maxidx(idx)) = out_maxval(out_maxidx(idx));
end

end