% shallow or 2-layer Artifical Neural Network
% part-2
% The code below is used to generate the regession plot at specific
% neuron.


% clear workspace and console
clear
clc

% For reproducibility
rng(13)                         

% load data
data = readmatrix("Data.txt");
X = data(1:20,1:3).';
Y = data(1:20,4).';

N = 6;                            % Neuron Unit

net = feedforwardnet(N);

net.trainFcn = 'trainlm';
                                  %  TrainFcn
                                  %  trainlm
                                  %  traincgb, traincgp, traingda,trainbfg
                                  %  traingdx, trainoss,trainscg,traingdm

net.layers{1}.transferFcn='logsig';
                                  %  transferFc
                                  %  logsig,purelin,tansig

% Train
[trainednet,tr] = train(net,X,Y);

% UI
nntraintool

% Y_pred = trainednet(X);
% mdl = fitlm(Y,Y_pred,'linear');
% disp(['R square is = ',num2str(mdl.Rsquared.Ordinary)]);
% disp(['Root Mean Sq Error is = ',num2str(mdl.RMSE)]);

% Saving....
% getNet = trainednet;
% save getNet
% disp('Net Saved!')
