% ANFIS

% The code below iterate the values of ClusterInfluenceRange
% and SquashFactor.The results contains ClusterInfluenceRange, 
% SquashFactor, Accept ratio, reject ratio, rules, R squared and
% Root mean square error(RMSE).


% Clear workspace and console
clear
clc


% Load Data
data = readmatrix('Data.txt');
X = data(:,1:3);
Y = data(:,4);

% Initialize
Results = zeros(10,7);
l = 0;

% Constants
AR = 0.5;   %Accept Ratio
RR = 0.15;  %Reject Ratio

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 100:5:200
    
    opt = genfisOptions('SubtractiveClustering');
    
    opt.AcceptRatio = AR;
    opt.RejectRatio = RR;
    
    % iteratation of value
    opt.SquashFactor = i/100; % range 1 to 1.5
    
    for j = 1:1:10
        
        % iteratation of value
        opt.ClusterInfluenceRange = j/10;% range .1 to 1
        
        % Generate Model
        fis = genfis(X,Y,opt);
        
        % Prediction
        Y_pred = evalfis(fis,X);
        
        %linear model
        mdl = fitlm(Y,Y_pred,'linear');
        
        % Appending the values
        Results(j+l,1) = j/10;
        Results(j+l,2) = i/100;
        Results(j+l,3) = AR;
        Results(j+l,4) = RR;
        Results(j+l,5) = length(fis.Rules);
        Results(j+l,6) = mdl.Rsquared.Ordinary;
        Results(j+l,7) = mdl.RMSE;
        
    end
    
    l = length(Results);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% temp  = array2table(Results,'VariableNames',{'Cluster Influence range',...
%         'Squash factor','Accepted Ratio','Rejected Ratio','Rules','R2','RMSE'});
% writetable(temp,'fuzzyResults.xlsx')
