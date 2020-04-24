function [fshared,lis]=make_obs_180420(fblocks,lr1,lr2,N,hhloc,maxhh,lrtol,ethcor)

%ASSIGN HOUSEHOLDS TO FEEDING BLOCKS. 
fgroups=prod(fblocks);
fu=N/fgroups;
fgrid1=sum(hhloc(:,1)*ones(1,fblocks(1))>ones(maxhh,1)*(1:fblocks(1))/fblocks(1),2);
fgrid2=sum(hhloc(:,2)*ones(1,fblocks(2))>ones(maxhh,1)*(1:fblocks(2))/fblocks(2),2);
fnum=(fgrid2)*fblocks(1)+fgrid1+1;
%Create matrix of shared feeding blocks at the household level
fshared=(fnum*ones(size(fnum))'==ones(size(fnum))*fnum')-eye(maxhh);

%COMPUTE INTERACTION RATES AMONG HOUSEHOLDS. 
%Make distance matrix between households
hhdm=sqrt((hhloc(:,1)*ones(1,maxhh)-ones(maxhh,1)*hhloc(:,1)').^2+...
    (hhloc(:,2)*ones(1,maxhh)-ones(maxhh,1)*hhloc(:,2)').^2);
%Compute the lens of shared area between households, scaled for the density
%with which individuals occupy that lens (which decreases in proportion to
%the area of the lens. Here, poeij is the density overlap between
%individuals using radii i and j, respectively (but poeij=poeji).
poe11=real(2*(lr1^2)*acos(hhdm./(2*lr1))-(1/2)*hhdm.*sqrt((2*lr1-hhdm).*(2*lr1+hhdm)))/(pi^2*lr1^4);
poe11=poe11-diag(diag(poe11));
poe22=real(2*(lr2^2)*acos(hhdm./(2*lr2))-(1/2)*hhdm.*sqrt((2*lr2-hhdm).*(2*lr2+hhdm)))/(pi^2*lr2^4);
poe22=poe22-diag(diag(poe22));
poe12=real(lr1^2*acos((hhdm.^2+lr1^2-lr2^2)./(2*hhdm*lr1))+lr2^2*acos((hhdm.^2+lr2^2-lr1^2)./(2*hhdm*lr2))...
    -(1/2)*sqrt((-hhdm+lr1+lr2).*(hhdm+lr1-lr2).*(hhdm-lr1+lr2).*(hhdm+lr1+lr2)))/(pi^2*lr1^2*lr2^2);
poe12(1:1+size(poe12,1):end) = 0;
%Scale to standard by which 2 people occupying the same circle with radius
%lrtol have one interaction per day. Store in a 3-D array.
lis=(pi*lrtol^2)*cat(3,poe11,poe12,poe22);
%Scale for reduced interaction rates among ethnicities.
lis=lis.*cat(3,ethcor,ethcor,ethcor);

% %SET LOCAL REGIONAL AND GLOBAL TRANSMISSION PROBABILITIES. This computes
% %the probability of transmission per encounter times the number of
% %encounters between individuals per day
% tl=aip*tr;                  %local
% tt=aip*tr*(2*3)/tu;         %toilet - assume possible transmission to person before or after
%                             %in line, 3 times per day
% tg=aip*tr*(2*3*(1/4))/fu;   %global - assume possible transmission to person before or after
%                             %in line, 3 times per day, but 1/4 of people go
