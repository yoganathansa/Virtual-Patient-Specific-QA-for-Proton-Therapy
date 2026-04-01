% -------------------------------------------------------------------------
% Project: Virtual Patient-Specific QA for Proton Therapy
%
% Author: Yoganathan
% Title: Medical Physicist
% Institution: Saint John Regional Hospital
% Location: Saint John, NB, Canada
%
% Description:
% This script is part of a framework for predicting measurement fluence
% from Treatment Planning System (TPS) data for virtual patient-specific QA.
%
% Disclaimer:
% This code is for research purposes only and is not intended for clinical use.
%
% Year: 2026
% -------------------------------------------------------------------------

%% Create Datastores for Training, Validation, and Testing
% clear all
addpath 'Y:\Matlab_Addon_Tools\imshow3Dfull.m'
addpath  Y:\Matlab_Addon_Tools\imoverlay\
addpath  Y:\Matlab_Addon_Tools\
reset(gpuDevice)

% training data
dstI = 'Y:\Gamma_Eval\Proton_PSQA\Train\TPS';
dstO =  'Y:\Gamma_Eval\Proton_PSQA\Train\Measurement';
imdsInp = imageDatastore(dstI,'FileExtensions','.mat',...
    'ReadFcn',@(x)load(x).tps_slice);
imdsRef = imageDatastore(dstO,'FileExtensions','.mat',...
    'ReadFcn',@(x)load(x).meas_slice);

% validation data
dstI = 'Y:\Gamma_Eval\Proton_PSQA\Test\TPS';
dstO =  'Y:\Gamma_Eval\Proton_PSQA\Test\Measurement';
imdsValInp = imageDatastore(dstI,'FileExtensions','.mat',...
    'ReadFcn',@(x)load(x).tps_slice);
imdsValRef = imageDatastore(dstO,'FileExtensions','.mat',...
    'ReadFcn',@(x)load(x).meas_slice);

% Combine input and response datastores
dsTrain = combine(imdsInp, imdsRef);
dsVal = combine(imdsValInp, imdsValRef);


%% Create minibatchqueue from the combined datastore
miniBatchSize = 16;

mbqTrain = minibatchqueue(dsTrain,...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@(x,y) deal(cat(4,x{:}), cat(4,y{:})),...
    PartialMiniBatch="discard",...
    MiniBatchFormat=["SSCB","SSCB"],...
    OutputEnvironment="gpu");

% Create minibatchqueue from the combined datastore
mbqVal = minibatchqueue(dsVal,...
    MiniBatchSize=3,...
    MiniBatchFcn=@(x,y) deal(cat(4,x{:}), cat(4,y{:})),...
    PartialMiniBatch="discard",...
    MiniBatchFormat=["SSCB","SSCB"],...
    OutputEnvironment="gpu");

%% New network
inputSize = [240 240 1];
intNumFilt = 32;

genX2Y = createEncoderDecoder2D(inputSize,intNumFilt,3,9,1,1); % (imageSize,initialNumFilters,numDownSamp,numResBlock,add_atten,numOPChannel)

%Create discriminators
discScale1 = PatchGAN2D([inputSize(1) inputSize(2) 3]);
discScale2 = PatchGAN2D([inputSize(1)/2 inputSize(2)/2 3]);

%% Load VGG Net for 2D perceptual loss
netVGGTF = imagePretrainedNetwork("vgg19");
netVGGTF = dlnetwork(netVGGTF.Layers(1:38));
% Convert to layer graph (needed for replaceLayer)
lgraph = layerGraph(netVGGTF);
% Replace the input layer with new size
inp = imageInputLayer([inputSize(1), inputSize(2), 3], ...
    "Normalization", "none", "Name", "unet_input");
lgraph = replaceLayer(lgraph, "input", inp);
% Convert to dlnetwork
netVGG = dlnetwork(lgraph);
%% Specify Training Options
trailAvgGen = []; trailAvgSqGen = [];
trailAvgDiscSc1 = []; trailAvgSqDiscSc1 = [];
trailAvgDiscSc2 = []; trailAvgSqDiscSc2 = [];
gradDecayFac = 0.5; sqGradDecayFac = 0.999;

dataDir = fullfile('Y:\Gamma_Eval\Proton_PSQA\');
%% Training Model
epoch = 0; iteration = 0; start = tic;
numEpochs =350;
% learning Rate parameters
initialLearnRate = 0.0008;
LearnRate = initialLearnRate;
decayFactor = 0.5;       % Reduce LR by N% every decay step
decayEveryEpochs = 25;   % Decay every N epochs
updateLR = decayEveryEpochs;
% validation parameters
ValLoss=100;
validationFrequency = 100;        % Check validation every 25 iterations

% Create a directory to store checkpoints
checkpointDir = fullfile(dataDir,"Models");
if ~exist(checkpointDir,"dir")
    mkdir(checkpointDir);
end

% monitor trainging progress
monitor = trainingProgressMonitor(...
    Metrics="Loss", ...
    Info=["LearningRate","Epoch","Iteration"], ...
    XLabel="Iteration");

numObservationsTrain = numel(dsTrain.UnderlyingDatastores{1, 1}.Files);
numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

% Initialize plots for training progress
[figureHandle, tileHandle, imageAxes1, imageAxes2, imageAxes3, scoreAxesY, LossAxesY, ...
    lineScoreGenXToY, lineScoreDiscY, lineValLossX2Y] = initializeTrainingPlot_SupGAN();
% Make sure figure is visible (especially in Live Script)
set(figureHandle, 'Visible', 'on');
executionEnvironment = "gpu";


while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Reset and shuffle the data
    reset(mbqTrain);
    shuffle(mbqTrain);

    % Loop over mini-batches
    while hasdata(mbqTrain) && ~monitor.Stop
        iteration = iteration + 1;

        % Read mini-batch of data
        [imageT,imageM] = next(mbqTrain);
   
        % Calculate the loss and gradients
        [gradParamsG, gradParamsDScale1, gradParamsDScale2, Loss, scores] = ...
            dlfeval(@modelGradients, imageT, imageM, genX2Y, discScale1, discScale2, netVGG);

        % Update parameters of generator
        [genX2Y, trailAvgGen, trailAvgSqGen] = adamupdate( ...
            genX2Y, gradParamsG, trailAvgGen, trailAvgSqGen, iteration, ...
            LearnRate, gradDecayFac, sqGradDecayFac);

        % Update parameters of discriminator scale1
        [discScale1, trailAvgDiscSc1, trailAvgSqDiscSc1] = adamupdate( ...
            discScale1, gradParamsDScale1, trailAvgDiscSc1, trailAvgSqDiscSc1, iteration, ...
            LearnRate, gradDecayFac, sqGradDecayFac);

        % Update the discriminator scale2 parameters
        [discScale2, trailAvgDiscSc2, trailAvgSqDiscSc2] = adamupdate( ...
            discScale2, gradParamsDScale2, trailAvgDiscSc2, trailAvgSqDiscSc2, iteration, ...
            LearnRate, gradDecayFac, sqGradDecayFac);
        
        % Update the plots of network scores and loss
        addpoints(lineScoreGenXToY, iteration, double(gather(extractdata(scores{1}))));
        addpoints(lineScoreDiscY, iteration, double(gather(extractdata(scores{2}))));
        legend(scoreAxesY, 'Gen','Disc');

        drawnow;

        % Update the title with training progress information.
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        title( "Ep: " + epoch + ", " +  "Iter: " + iteration + ", " + ...
            "ValLoss: " + ValLoss + "," + "Elap: " + string(D))

        % Update the training progress - Record metrics.
        recordMetrics(monitor,iteration,Loss=Loss);
        updateInfo(monitor,LearningRate=LearnRate,Epoch=epoch,Iteration=iteration);
        monitor.Progress = 100 * iteration/numIterations;

        % Display Validation images
        if mod(iteration,validationFrequency) == 0 || iteration == 1
            ValLoss = displayGeneratedValImages_SupGAN(mbqVal, imageAxes1, imageAxes2, imageAxes3,...
                LossAxesY, lineValLossX2Y, genX2Y, iteration);
            drawnow;
        end

        % --- Decay learning rate every decayEveryEpochs ---
        if mod(epoch, updateLR) == 0
            LearnRate = LearnRate * decayFactor;
            % fprintf('Epoch %d: learning rate decayed to %g\n', epoch, LearnRate);
            updateLR = updateLR + decayEveryEpochs;
        end
    end
end
% Save the final model
modeEateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
save(checkpointDir+filesep+"PSQA_Pred-"+modeEateTime+".mat", ...
    'genX2Y','discScale1',"discScale2");
%% Generate New Images Using Test Data
load("Y:\Gamma_Eval\Proton_PSQA\Models\PSQA_Pred-2025-12-11_BEST.mat")
% validation data
dstI = 'Y:\Gamma_Eval\Proton_PSQA\Test\TPS';
dstO =  'Y:\Gamma_Eval\Proton_PSQA\Test\Measurement';
imdsValInp = imageDatastore(dstI,'FileExtensions','.mat',...
    'ReadFcn',@(x)load(x).tps_slice);
imdsValRef = imageDatastore(dstO,'FileExtensions','.mat',...
    'ReadFcn',@(x)load(x).meas_slice);

for idx = 1:length(imdsValRef.Files)
    imageTestT = read(imdsValInp);
    imageTestM = read(imdsValRef);
    TPS_Image(:,:,idx) = imageTestT;
    Meas_Image(:,:,idx) = imageTestM;

    % Convert mini-batch of data to dlarray and specify the dimension labels
    % "SSCB" (spatial, spatial, channel, batch)
    imageTestT = dlarray(imageTestT,"SSCB");
    imageTestM = dlarray(imageTestM,"SSCB");

    % If running on a GPU, then convert data to gpuArray
    if canUseGPU
        imageTestT = gpuArray(imageTestT);
        imageTestM = gpuArray(imageTestM);
    end

    % Generate translated images
    genImageM(:,:,idx) = extractdata(gather(predict(genX2Y,imageTestT)));
end
clear tps_gamma measured_gamma
% Resize to 1mm
for i=1:size(TPS_Image,3)
    Ref_Test(:,:,i) = imresize(Meas_Image(:,:,i),[243 243]);
    Tps_Test(:,:,i) = imresize(TPS_Image(:,:,i),[243 243]);
    Pred_Test(:,:,i) = imresize(genImageM(:,:,i),[243 243]);
end

% Normalize the images
for i=1:size(TPS_Image,3)
    Ref_Test(:,:,i) = rescale(Ref_Test(:,:,i),0,1);
    Tps_Test(:,:,i) = rescale(Tps_Test(:,:,i),0,1);
    Pred_Test(:,:,i) = rescale(Pred_Test(:,:,i),0,1);
end
% mask background
mask = zeros(size(Tps_Test)); mask(Tps_Test>0.05) = 1;
Ref_Test = Ref_Test.*mask;
Pred_Test = Pred_Test.*mask;
Tps_Test = Tps_Test.*mask;

%% Gamma evaluation
%3%/3mm
dose_diff = 3; dth = 3; tol = 10;
for i=1:size(Tps_Test,3)
    % Global gamma value
    [Gamma_GlobVal33(i,1), Gamma_GlobMat_ref33(:,:,i)] = myGammaGlobal(double(Ref_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    [Gamma_GlobVal33(i,2),Gamma_GlobMat_pred33(:,:,i)] = myGammaGlobal(double(Pred_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    % Local gamma value
    [Gamma_LocVal33(i,1),Gamma_LocMat_ref33(:,:,i)] = myGammaLocal(double(Ref_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    [Gamma_LocVal33(i,2),Gamma_LocMat_pred33(:,:,i)] = myGammaLocal(double(Pred_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
end
%3%/2mm
dose_diff = 3; dth = 2; tol = 10;
for i=1:size(Tps_Test,3)
    % Global gamma value
    [Gamma_GlobVal32(i,1), Gamma_GlobMat_ref32(:,:,i)] = myGammaGlobal(double(Ref_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    [Gamma_GlobVal32(i,2),Gamma_GlobMat_pred32(:,:,i)] = myGammaGlobal(double(Pred_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    % Local gamma value
    [Gamma_LocVal32(i,1),Gamma_LocMat_ref32(:,:,i)] = myGammaLocal(double(Ref_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    [Gamma_LocVal32(i,2),Gamma_LocMat_pred32(:,:,i)] = myGammaLocal(double(Pred_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
end
% 2%/2mm
dose_diff = 2; dth = 2; tol = 10;
for i=1:size(Tps_Test,3)
    % Global gamma value
    [Gamma_GlobVal22(i,1), Gamma_GlobMat_ref22(:,:,i)] = myGammaGlobal(double(Ref_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    [Gamma_GlobVal22(i,2),Gamma_GlobMat_pred22(:,:,i)] = myGammaGlobal(double(Pred_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    % Local gamma value
    [Gamma_LocVal22(i,1),Gamma_LocMat_ref22(:,:,i)] = myGammaLocal(double(Ref_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
    [Gamma_LocVal22(i,2),Gamma_LocMat_pred22(:,:,i)] = myGammaLocal(double(Pred_Test(:,:,i)*100),double(Tps_Test(:,:,i)*100),dose_diff,dth,tol);
end

%% Plot Gamma Distribution

% Determine Best, Median and Worst
 clc;

saveFolder = 'Y:\Gamma_Eval\Proton_PSQA\Imgs';

% Load values
Gamma_LocVal22; Gamma_LocVal32; Gamma_LocVal33;
Gamma_GlobVal22; Gamma_GlobVal32; Gamma_GlobVal33;

GammaLocVals  = {Gamma_LocVal22, Gamma_LocVal32, Gamma_LocVal33};
GammaGlobVals = {Gamma_GlobVal22, Gamma_GlobVal32, Gamma_GlobVal33};

labels = {'2%/2mm','3%/2mm','3%/3mm'};
titles = {'Best','Median','Worst'};

% Load maps
Gamma_LocMat_ref22; Gamma_LocMat_ref32; Gamma_LocMat_ref33;
Gamma_GlobMat_ref22; Gamma_GlobMat_ref32; Gamma_GlobMat_ref33;

RefLoc  = {Gamma_LocMat_ref22, Gamma_LocMat_ref32, Gamma_LocMat_ref33};
RefGlob = {Gamma_GlobMat_ref22, Gamma_GlobMat_ref32, Gamma_GlobMat_ref33};

PredLoc = {Gamma_LocMat_pred22, Gamma_LocMat_pred32, Gamma_LocMat_pred33};
PredGlob = {Gamma_GlobMat_pred22, Gamma_GlobMat_pred32, Gamma_GlobMat_pred33};

% Plot settings
cmap = parula;
minVal = 0;
maxVal = 2;

% FIGURE 1 → LOCAL

figure('Color','w','Position',[100 100 900 900]);

t = tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
sgtitle('Gamma Evaluation - LOCAL','FontWeight','bold');

for i = 1:3
    
    col1 = GammaLocVals{i}(:,1);
    col2 = GammaLocVals{i}(:,2);
    
    diff = abs(col1 - col2);
    [~, sortIdx] = sort(diff);
    
    n = length(diff);
    idxSet = [sortIdx(1), sortIdx(round(n/2)), sortIdx(end-1)];
    
    for j = 1:3
        
        idx = idxSet(j);
        nexttile;
        
        % Combine Reference + Prediction
        img = [RefLoc{i}(:,:,idx), PredLoc{i}(:,:,idx)];
        imagesc(img, [minVal maxVal]);
        axis image off;
        
        % Titles (top row only for clean look)
        if i == 1
            title(titles{j},'FontWeight','bold');
        end
        
        % Row labels (left side only)
        if j == 1
            ylabel(labels{i},'FontWeight','bold');
        end
        
        % Add pass rate text inside image
        % text(10,20, ...
        %     ['Ref: ' sprintf('%.1f', col1(idx)) ...
        %     '  Pred: ' sprintf('%.1f', col2(idx))], ...
        %     'Color','w','FontSize',9,'FontWeight','bold');

        w = size(RefLoc{i},2);   % width of one image

        % --- Reference (LEFT) ---
        text(15,20, ...
            ['Reference GPR (%): ' sprintf('%.1f', col1(idx))], ...
            'Color','w','FontSize',9,'FontWeight','bold');

        % --- Prediction (RIGHT) ---
        text(w + 15,20, ...
            ['Prediction GPR (%): ' sprintf('%.1f', col2(idx))], ...
            'Color','w','FontSize',9,'FontWeight','bold');
        
        % Vertical separator
        hold on;
        xline(size(RefLoc{i},2)+0.5,'w','LineWidth',1.2);
        hold off;
    end
end

% Shared colorbar
cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Gamma Value';

colormap(parula);
caxis([minVal maxVal]);

%% FIGURE 2 → GLOBAL

figure('Color','w','Position',[100 100 900 900]);

t = tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
sgtitle('Gamma Evaluation - GLOBAL','FontWeight','bold');

for i = 1:3
    
    col1 = GammaLocVals{i}(:,1);
    col2 = GammaLocVals{i}(:,2);
    
    diff = abs(col1 - col2);
    [~, sortIdx] = sort(diff);
    
    n = length(diff);
    idxSet = [sortIdx(1), sortIdx(round(n/2)), sortIdx(end-1)];

    col1 = GammaGlobVals{i}(:,1);
    col2 = GammaGlobVals{i}(:,2);
    
    for j = 1:3
        
        idx = idxSet(j);
        nexttile;
        
        img = [RefGlob{i}(:,:,idx), PredGlob{i}(:,:,idx)];
        imagesc(img, [minVal maxVal]);
        axis image off;
        
        if i == 1
            title(titles{j},'FontWeight','bold');
        end
        
        if j == 1
            ylabel(labels{i},'FontWeight','bold');
        end
        
        % text(10,20, ...
        %     ['Ref: ' sprintf('%.1f', col1(idx)) ...
        %      '  Pred: ' sprintf('%.1f', col2(idx))], ...
        %      'Color','w','FontSize',9,'FontWeight','bold');

        % --- Reference (LEFT) ---
        text(15,20, ...
            ['Reference GPR (%): ' sprintf('%.1f', col1(idx))], ...
            'Color','w','FontSize',9,'FontWeight','bold');

        % --- Prediction (RIGHT) ---
        text(w + 15,20, ...
            ['Prediction GPR (%): ' sprintf('%.1f', col2(idx))], ...
            'Color','w','FontSize',9,'FontWeight','bold');
        
        hold on;
        xline(size(RefGlob{i},2)+0.5,'w','LineWidth',1.2);
        hold off;
    end
end

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Label.String = 'Gamma Value';

colormap(parula);
caxis([minVal maxVal]);

% Save (optional)
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% exportgraphics(gcf, fullfile(saveFolder,'Gamma_LOCAL.png'),'Resolution',300);

%% Supporting Functions

% GAN Model Gradients Function

function [gradParamsG, gradParamsDScale1, gradParamsDScale2, lossGCombined, scores] = ...
    modelGradients(input, realImg, generator, disc1, disc2, netVGG)

% ---- Forward generator once ----
genImg = forward(generator,input);

% calculate gamma evaluation
b = randperm(size(input,10));
[RefGammaVal,RefGammaMap] = myGamma_gpu_batch(realImg, input, 2, 2, 10, b, 'local');
[PredGammaVal,PredGammaMap] = myGamma_gpu_batch(genImg, input, 2, 2, 10, b, 'local');

% Logical mask of failing pixels across all slices
failMask = (RefGammaMap > 1) | (PredGammaMap > 1);
gammaDiff = abs(RefGammaMap(failMask) - PredGammaMap(failMask));
GammaLoss = (mean(abs(RefGammaVal-PredGammaVal),'all') + mean(gammaDiff,'all'))*10;

% Scale 1
[DL1, GL1, real1D, fake1G, GSc1, DSc1] = pix2pixHDAdverserialLoss(realImg,genImg,disc1);

% Scale 2
rsR = dlresize(realImg,Scale=0.5,Method="linear");
rsG = dlresize(genImg,Scale=0.5,Method="linear");
[DL2, GL2, real2D, fake2G, GSc2, DSc2] = pix2pixHDAdverserialLoss(rsR,rsG,disc2);

% Feature matching
FML1 = pix2pixHDFeatureMatchingLoss(real1D, fake1G) * 5;
FML2 = pix2pixHDFeatureMatchingLoss(real2D, fake2G) * 5;

% VGG
rRGB = repmat(realImg,[1 1 3 1]);
gRGB = repmat(genImg,[1 1 3 1]);
VGG  = pix2pixHDVGGLoss(rRGB,gRGB,netVGG) * 5;

scores = {GSc1+GSc2, DSc1+DSc2};

% ---- Final combined generator loss ----
lossGCombined = GL1 + GL2 + FML1 + FML2 + VGG + GammaLoss;

% ---- Gradients ----
gradParamsG = dlgradient(lossGCombined, generator.Learnables, RetainData=true);

lossD = 0.5*(DL1 + DL2);
gradParamsDScale1 = dlgradient(lossD, disc1.Learnables, RetainData=true);
gradParamsDScale2 = dlgradient(lossD, disc2.Learnables);

end

%%%%%%%% ADVERSARIAL LOSS %%%%%%%%%%%%%%%%%%
function [DLoss,GLoss,realPredFtrsD,genPredFtrsD,GScore,DScore] = pix2pixHDAdverserialLoss(inpReal,inpGenerated,discriminator)

% Convert to RGB by repeating channels
inpReal = repmat(inpReal, [1 1 3 1]);
inpGenerated   = repmat(inpGenerated, [1 1 3 1]);

% Discriminator layer names containing feature maps
featureNames = ["act_top","act_mid_1","act_mid_2","act_tail","conv2d_final"];

% Get the feature maps for the real image from the discriminator
realPredFtrsD = cell(size(featureNames));
[realPredFtrsD{:}] = forward(discriminator,inpReal,Outputs=featureNames);

% Get the feature maps for the generated image from the discriminator
genPredFtrsD = cell(size(featureNames));
[genPredFtrsD{:}] = forward(discriminator,inpGenerated,Outputs=featureNames);

% Get the feature map from the final layer to compute the loss
realPredD = realPredFtrsD{end};
genPredD = genPredFtrsD{end};

% Calculate scores of generators
GScore = mean(sigmoid(genPredD),"all");

% Calculate Score - Discriminator
DScore = 0.5*mean(sigmoid(realPredD),"all") + ...
    0.5*mean(1-sigmoid(genPredD),"all");

% Compute the discriminator loss
DLoss = (1 - realPredD).^2 + (genPredD).^2;
DLoss = mean(DLoss,"all");

% Compute the generator loss
GLoss = (1 - genPredD).^2;
GLoss = mean(GLoss,"all");
end

%%%%%%%%%%%% FEATURE MATCHING LOSS %%%%%%%%%%%%%%%%%%%%%
function featureMatchingLoss = pix2pixHDFeatureMatchingLoss(realPredFtrs,genPredFtrs)

% Number of features
numFtrsMaps = numel(realPredFtrs);

% Initialize the feature matching loss
featureMatchingLoss = 0;

for i = 1:numFtrsMaps
    % Get the feature maps of the real image
    a = extractdata(realPredFtrs{i});
    % Get the feature maps of the synthetic image
    b = genPredFtrs{i};

    % Compute the feature matching loss
    featureMatchingLoss = featureMatchingLoss + mean(abs(a - b),"all");
end
end

%%%%%%% VGG LOSS ############
function vggLoss = pix2pixHDVGGLoss(realImage,generatedImage,netVGG)

featureWeights = [1.0/32 1.0/16 1.0/8 1.0/4 1.0];

% Initialize the VGG loss
vggLoss = 0;

% Specify the names of the layers with desired feature maps
featureNames = ["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"];

% Extract the feature maps for the real image
activReal = cell(size(featureNames));
[activReal{:}] = forward(netVGG,realImage,Outputs=featureNames);

% Extract the feature maps for the synthetic image
activGenerated = cell(size(featureNames));
[activGenerated{:}] = forward(netVGG,generatedImage,Outputs=featureNames);

% Compute the VGG loss
for i = 1:numel(featureNames)
    vggLoss = vggLoss + featureWeights(i)*mean(abs(activReal{i} - activGenerated{i}),"all");
end
end
%% Validation Images display
function  ValLoss = displayGeneratedValImages_SupGAN(mbq, imageAxes1, imageAxes2, imageAxes3,...
    LossAxesY, lineValLossX2Y, genXToY, iteration)
% displayGenerated validation Images Displays generated images

% Read validation data
if hasdata(mbq) == 0
    reset(mbq);
    shuffle(mbq);
end

% Read mini-batch of data
[imX,imY] = next(mbq);

% Generate images using the held-out generator input.
imYGenerated = predict(genXToY,imX);

% Extract image data
realImages = extractdata(imY);
generatedImages = extractdata(imYGenerated);
InpImages = extractdata(imX);

Ref1 = gather(squeeze(realImages)); Ref = zeros(243,243);
TPS1 = gather(squeeze(InpImages)); TPS = zeros(243,243);
Pred1 = gather(squeeze(generatedImages)); Pred = zeros(243,243);

for i=1:size(TPS1,3)
    Ref(:,:,i) = imresize(Ref1(:,:,i),[243 243]);
    TPS(:,:,i) = imresize(TPS1(:,:,i),[243 243]);
    Pred(:,:,i) = imresize(Pred1(:,:,i),[243 243]);
end
% Local gamma value
b = randperm(size(InpImages,4));
[RefGamma,~] = myGamma_gpu_batch(double(Ref),double(TPS),3,2,10,b,'local');
[PredGamma,~] = myGamma_gpu_batch(double(Pred),double(TPS),3,2,10,b,'local');

ValLoss = mean(abs(RefGamma - PredGamma),'all');

% create 3 random index
b = randperm(size(imY,4), 3);
% ---- Plot first Image ----
cla(imageAxes1);
imageResults = gather(cat(2, realImages(:,:,b(1)), generatedImages(:,:,b(1))));
imshow(imageResults, [], 'Parent', imageAxes1);
title(imageAxes1, "Real (Left) vs Generated (Right)");

% ---- Plot second Image ----
cla(imageAxes2);
imageResults = gather(cat(2, realImages(:,:,b(2)), generatedImages(:,:,b(2))));
imshow(imageResults, [], 'InitialMagnification', 'fit', 'Parent', imageAxes2);
title(imageAxes2, "Real (Left) vs Generated (Right)");

% ---- Plot third Image ----
cla(imageAxes3);
imageResults = gather(cat(2, realImages(:,:,b(3)), generatedImages(:,:,b(3))));
imshow(imageResults, [], 'InitialMagnification', 'fit', 'Parent', imageAxes3);
title(imageAxes3, "Real (Left) vs Generated (Right)");

%update plot
addpoints(lineValLossX2Y,iteration,double(ValLoss))
legend(LossAxesY, 'ValLoss');

drawnow;
end


%% Initilialize figure
function [figureHandle, tileHandle, imageAxes1,imageAxes2,imageAxes3, scoreAxesY, LossAxesY, ...
    lineScoreGenXToY, lineScoreDiscY, lineValLossX2Y] = initializeTrainingPlot_SupGAN()
% Initialize figure layout for SupGAN training

% Create a wide figure for displaying training progress
figureHandle = figure('Name','Training Progress (SupGAN)');
figureHandle.Position(3) = 1.5 * figureHandle.Position(3); % Make figure wider

% Layout: M rows x N columns
tileHandle = tiledlayout(figureHandle, 2, 3, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');

% Tile 1: For image display 
imageAxes1 = nexttile(tileHandle, 1);
title(imageAxes1, 'Validation Slices');
axis off;

% Tile 2: For image display 
imageAxes2 = nexttile(tileHandle, 2);
title(imageAxes2, 'Validation Slices');
axis off;

% Tile 3: For image display 
imageAxes3 = nexttile(tileHandle, 3);
title(imageAxes3, 'Validation Slices');
axis off;

% Tile 4: Validation loss
LossAxesY = nexttile(tileHandle, 4);
xlabel(LossAxesY, "Iteration");
ylabel(LossAxesY, "Loss");
grid(LossAxesY, "on");


% Tile 3: Generator & Discriminator scores
scoreAxesY = nexttile(tileHandle, 6);
xlabel(scoreAxesY, "Iteration");
ylabel(scoreAxesY, "Score");
grid(scoreAxesY, "on");


% Initialize animated lines
lineScoreGenXToY = animatedline(scoreAxesY, 'Color', 'r', 'LineWidth', 1.5);
lineScoreDiscY   = animatedline(scoreAxesY, 'Color', 'g', 'LineWidth', 1.5);
lineValLossX2Y   = animatedline(LossAxesY, 'Color', 'b', 'Marker', '*', 'LineWidth', 1.2);

end
%%
function [pass_rate, gammaMap] = myGamma_gpu_batch(doseEval, doseRef, dDose, dDistance, Threshold, sliceIdx, mode)
% ============================================================
% Fully vectorized TG-218 compliant gamma (GPU)
% Supports LOCAL and GLOBAL gamma
%
% Inputs:
%   doseEval  - evaluated dose (HxWxDxB or dlarray)
%   doseRef   - reference dose (HxWxDxB or dlarray)
%   dDose     - dose difference criterion (%) e.g., 3
%   dDistance - distance to agreement criterion (pixels)
%   Threshold - pixels >= Threshold (% of maxRef)
%   sliceIdx  - slice indices to process
%   mode      - 'local' or 'global'
%
% Outputs:
%   pass_rate   - gamma pass rate (%)
%   gammaMatrix - gamma map
% ============================================================

% Convert dlarray to GPU array
if isa(doseEval,'dlarray'), doseEval = gpuArray(extractdata(doseEval)); end
if isa(doseRef,'dlarray'), doseRef = gpuArray(extractdata(doseRef)); end

pass_rate = zeros(1,length(sliceIdx));

for bb = 1:length(sliceIdx)
    doseE = squeeze(double(doseEval(:,:,sliceIdx(bb))));
    doseR = squeeze(double(doseRef(:,:,sliceIdx(bb))));
    [rows, cols] = size(doseE);

    % Normalize to [0,100] and max reference 
    doseE = ((doseE + 1) / 2) * 100;
    doseR = ((doseR + 1) / 2) * 100;
    maxRef = max(doseR(:));
    doseE = doseE * 100 / maxRef;
    doseR = doseR * 100 / maxRef;

    % Threshold mask 
    evalMask = doseE >= Threshold;

    % Precompute neighbor shifts for incremental perimeter search
    neighborShifts = [];
    for k = 1:dDistance
        for dx = -k:k
            for dy = -k:k
                if abs(dx)==k || abs(dy)==k
                    neighborShifts = [neighborShifts; dx, dy];
                end
            end
        end
    end
    numShifts = size(neighborShifts,1);

    % Initialize gamma map 
    gammaMatrix = gpuArray(-ones(rows,cols));
    gammaVals = gpuArray(Inf(rows,cols,numShifts));

    % Compute gamma for each shift vectorized
    for s = 1:numShifts
        dx = neighborShifts(s,1);
        dy = neighborShifts(s,2);

        % Shift reference
        shiftedRef = circshift(doseR, [dy, dx]);

        % Dose difference term
        switch lower(mode)
            case 'global'
                denom = (dDose/100 * maxRef)^2 + eps;       % global denominator
            case 'local'
                denom = (dDose/100 * doseE).^2 + eps;      % local denominator per pixel
            otherwise
                error('Mode must be ''local'' or ''global''');
        end
        doseTerm = (doseE - shiftedRef).^2 ./ denom;

        % Spatial term (normalized squared distance)
        spatialTerm = (dx^2 + dy^2) / (dDistance^2);

        gammaVals(:,:,s) = doseTerm + spatialTerm;
    end

    % Minimum gamma over all shifts
    minGammaSq = min(gammaVals,[],3);
    gammaMatrix(evalMask) = sqrt(minGammaSq(evalMask));

    % Pass rate
    gCPU = gather(gammaMatrix);
    N_unEval = sum(gCPU(:)<0);
    N_fail = sum(gCPU(:)>1);
    totalEval = numel(gCPU) - N_unEval;
    pass_rate(bb) = 100*(1 - N_fail/totalEval);
    gammaMap(:,:,bb) = gammaMatrix;
end

end






