function [h] = H_FlowVecVisual(flw)
% flw: optical flow (vx,vy)

[nr,nc,~ ] = size(flw);
x = repmat(1:nc,nr,1);
y = repmat((1:nr)',1,nc);
quiver(x,y,flw(:,:,1),flw(:,:,2),0,'Color','r')
set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse')%set origin at top-left location
axis off