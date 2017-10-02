function [h,weight] = H_OpticalWeightVisual(w)
% load(fullfile('cifarCAE_2/net-epoch-50.mat'))
% load(fullfile('cifarCAE_3/net-epoch-6.mat'))
% load(fullfile('/usr/not-backed-up/matconvnet-1.0-beta14/examples/Oct2016Exp/OptUCSDMeanSub32dim/net-epoch-500.mat'))

%% Note: w has to be in [r*c*nch*nf] with nch = 3 or 1
% w = net.params(1).value;
w_ = w;

[filtS,~,nch,numfil] = size(w_);
r = ceil(sqrt(numfil));
c = ceil(numfil./r);

weight = zeros(r*filtS+r-1,c*filtS+c-1,size(w_,3));
for i = 1:size(w_,4)
    c_ = rem(i,c);
    if c_ == 0; c_ = c;end
    r_ = ceil(i/c);
    weight((filtS+1)*r_ - filtS:(filtS+1)*r_ -1,(filtS+1)*c_ - filtS:(filtS+1)*c_ -1,:) = w_(:,:,:,i);
end
[nr,nc,~ ] = size(weight);
x = repmat(1:nc,nr,1);
y = repmat((1:nr)',1,nc);
quiver(x,y,weight(:,:,1),weight(:,:,2),0,'Color','r')
set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse')%set origin at top-left location
axis off
% % figure
% if nch == 3
%     h = imshow(weight);
% elseif nch == 1
%     h = imagesc(weight);
% else 
%     disp('number channel of filters must 3 or 1')
% end
% axis image off
% drawnow;

% figure
% if nch == 3
%     imshow(weight)
% elseif nch == 1
%     imagesc(weight)
% else 
%     disp('number channel of filters must 3 or 1')
% end