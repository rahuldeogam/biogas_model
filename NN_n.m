% shallow or 2-layer Artifical Neural Network
% Part-1
% The code below iterate the values of neuron in a 
% feedforward net inside a loop. The maximum number
% of neurons is denoted by UL(upper limit) of the loop.
% The final results has neuron numbers, R squared and
% Root mean square error(RMSE).


% clear variables and console
clear
clc

% Initialize
Ul = 10;
Results = zeros(Ul,3);

% load data
data =  readmatrix("Data.txt");
X = data(1:20,1:3).';
Y = data(1:20,4).';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for N = 1:1:Ul
    
    % for reproducibility
    rng(13)
    
    % Iteration of neurons
    net = feedforwardnet(N);
    
    % Training function
    net.trainFcn = 'trainlm';
   
                                    %  TrainFcn exanple
                                    %  trainlm
                                    %  traincgb, traincgp, traingda,trainbfg
                                    %  traingdx, trainoss,trainscg,traingdm
    % Transfer Function                             
    net.layers{1}.transferFcn='tansig';
                                    %  transferFc
                                    %  logsig,purelin,tansig
    % Training
    [trainednet,tr] = train(net,X,Y);

    % Prediction   
    Y_pred = trainednet(X);
    % linear model
    mdl = fitlm(Y,Y_pred,'linear');
    
    
    % Appending the values
    Results(N,1)=N;                          % neurons
    Results(N,2) = mdl.Rsquared.Ordinary;    % Rsquared
    Results(N,3) = mdl.RMSE;                 % RMSE
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Convert the array into table and export it as "tainFcn_transferFnc.xlsx",
%eg:lm_tansig.xlsx
% temp  = array2table(Results,'VariableNames',{'Hidden Unit','Epoch','MSE'});
% writetable(temp,'lm_log.xlsx')

%plotting graph
figure
plot(Results(:,1),Results(:,3),'-o')
title('UnitsVsMse')
xlabel('Hidden Units')
ylabel('MSE')

% closing nntraintool UI
nntraintool close