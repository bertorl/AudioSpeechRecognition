%%
clear all
clc all
%%
load rawDataset.mat
load labelsNumbers.mat
labels = str2num(labels);
%% EXTRACCIÓN DE MFCC

fs = 8000;
win = hann(256,"periodic");
featuresTrain = {}; 
for i=1:length(labels)
    S = stft(rawDataset(i,:)',"Window",win,"OverlapLength",128,"Centered",false);
    y_train = mfcc(S,fs,"LogEnergy","Ignore");
    featuresTrain{i,1} = y_train;
end

%% DISEÑO DE LA RED
options = trainingOptions("adam", ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",false);

[numFeatures,numHopsPerSequence] = size(featuresTrain{1});
labelsTrain = labels;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(50,"OutputMode","last")
    fullyConnectedLayer(numel(unique(labelsTrain)))
    softmaxLayer
    classificationLayer];

%% TRAIN
labelsTrain = categorical(labelsTrain);
net = trainNetwork(featuresTrain,labelsTrain,layers,options);

%% TEST (OPCIONAL)
% Di un número en inglés
recObj = audiorecorder;
disp('Start speaking.'); recordblocking(recObj, 1); disp('End of Recording.');

dataTest = getaudiodata(recObj);
dataTest = [dataTest' zeros(1,9000-length(dataTest))];

S_Test = stft(dataTest',"Window",win,"OverlapLength",128,"Centered",false);
y_test = mfcc(S_Test,fs,"LogEnergy","Ignore");
featuresTest{1,1} = y_test;

classify(net,featuresTrain)




