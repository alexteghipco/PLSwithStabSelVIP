function [pls_yh,yNorm,winC,winCLoss,fk,allBetas] = plsCV(X,y,alpha,numComponentsRange,cvo,cvi,standardizeX,standardizeY,parallel,feats,ssVals)
% plsCV performs partial least squares regression in a nested CV scheme. 
% You must provide your own outer and inner cvpartitions. This will also
% interface with stability selection to run independent PLS with VIP selection
% on 50 complementary bootstraps of the training datasets 
% (outer folds; using 50% of the data). StabSel will be run across selected dimensionalities 
%
% INPUTS -----
% X: n x p matrix of n samples and p predictors
% y: n x r matrix of n samples and r response variables to predict
% alpha: weighting of correlation vs MSE (0 is all correlation, 1 is all MSE)
% numComponentsRange: full set of # of components to test in inner loop
% cvo: 1 x o cell of cvpartition objects for outer folds. If > 1 this amounts to repeats of CV
% cvi: o x i cell of cvpartition objects for inner folds. i reflects the inner folds per o (i.e., outer fold)
% standardizeX: true or false to do standardization of X or not (columnwise z-scoring)
% standardizeY: true or false to do standardization of y or not (columnwise z-scoring)
% parallel: if true, pls with use parallelization of cpus
% feats: set empty to ignore or give cell array where values within each cell correspond to the indices of a specific set of features (columns in X) to use during training
% ssVals: set empty to ignore or set to a value which will act as the threshold for VIP. Features > this VIP in bootstraps of the data will be "selected" by stability selection
%
% OUTPUTS -----
% pls_yh : predictions of y
% yNorm : normalized y values if standardization was true
% winC: winning # of components
% winCLoss: loss for winning # of components
% fk: selected features by stability selection with VIP
% allBetas: betas of outer training fold PLS models

tmpsave = false; % set this to true to save out intermediary data
saveBetas = true; % if true, betas will be saved
impute = true; % if data is missing, impute with KNN by taking data from closest sample based on correlation

%pls_yh = zeros(size(y));
if isempty(feats)
    feats = repmat({1:size(X,2)},length(cvo),length(cvi{1})); % cvo.NumTestSets
end

doSS = true;
if isempty(ssVals)
    ssVals = 1;
    doSS = false;
end
if doSS
    warning('Make sure you have my stabSel repository installed!!')
end

if impute
    X = knnimpute(X,1,'Distance','correlation'); % impute missing data based on closest columns (i.e., fill in missing values based on ROI that is most like missing ROI for a participant)
end
allBetas = [];
fk = [];

for r = 1:length(cvo)
    disp(['Repeat is ' num2str(r)])
    for i = 1:cvo{r}.NumTestSets
        disp(['-----Test set ' num2str(i) ' of ' num2str(cvo{r}.NumTestSets)])
        trX = X(cvo{r}.training(i),feats{r,i});
        trY = y(cvo{r}.training(i),:);
        teX = X(cvo{r}.test(i), feats{r,i});
        teY = y(cvo{r}.test(i), :);
        
        % Tune the number of components using the inner cvpartition
        bestCustomLoss = Inf;
        bestNumComponents = 0;
        bestSS = 0;
        
        mx = max([size(trX,2) size(trY,2)]);
        ncrtmp = numComponentsRange;
        idx = find(ncrtmp > mx);
        ncrtmp(idx) = [];
        err=false;

        % first perform stabsel for all folds
        for k = 1:cvi{r}{i}.NumTestSets
            trXX = trX(cvi{r}{i}.training(k), :);
            trYY = trY(cvi{r}{i}.training(k), :);
            teXX = trX(cvi{r}{i}.test(k), :);
            teYY = trY(cvi{r}{i}.test(k), :);
            
            if standardizeX
                [trXX,C,S] = normalize(trXX);
                teXX = normalize(teXX,'Center',C,'Scale',S);
            end
            if standardizeY
                [trYY,C2,S2] = normalize(trYY);
                teYY = normalize(teYY,'Center',C2,'Scale',S2);
            end
            
            if parallel
                if doSS
                    [fktmp{r,k},~,fscmxtmp{r,k},~,~,~,~,~,~,~,~,~,~,~] = stabSel(trXX,trYY,...
                        'parallel',false,'stnd',standardizeX,'compPars',true,'samType','bootstrap','filter',false,'filterThresh',1.5,...
                        'maxVars',[],'thresh',max(ssVals),'fdr',[],'numFalsePos',[],'nComp',[min(ncrtmp):1:max(ncrtmp)],'selAlgo',...
                        'pls','rep',50,'prop',0.5,'parallel',true,'verbose',false); %maxVars 50 gives 16
                end
            else
                if doSS
                     [fktmp{r,k},~,fscmxtmp{r,k},~,~,~,~,~,~,~,~,~,~,~] = stabSel(trXX,trYY,...
                        'parallel',false,'stnd',standardizeX,'compPars',true,'samType','bootstrap','filter',false,'filterThresh',1.5,...
                        'maxVars',[],'thresh',max(ssVals),'fdr',[],'numFalsePos',[],'nComp',[min(ncrtmp):1:max(ncrtmp)],'selAlgo',...
                        'pls','rep',50,'prop',0.5,'parallel',true,'verbose',false); %maxVars 50 gives 16
                end
            end
        end
        
        % now pass in features kept based on various stability thresholds
        % and use different numbers of components
        for w = ssVals
            for nc = ncrtmp
                custom_loss_sum = 0;
                for k = 1:cvi{r}{i}.NumTestSets
                    trXX = trX(cvi{r}{i}.training(k), :);
                    trYY = trY(cvi{r}{i}.training(k), :);
                    teXX = trX(cvi{r}{i}.test(k), :);
                    teYY = trY(cvi{r}{i}.test(k), :);
                    
                    if standardizeX
                        [trXX,C,S] = normalize(trXX);
                        teXX = normalize(teXX,'Center',C,'Scale',S);
                    end
                    if standardizeY
                        [trYY,C2,S2] = normalize(trYY);
                        teYY = normalize(teYY,'Center',C2,'Scale',S2);
                    end
                    
                    if doSS
                        idx = find(fscmxtmp{r,k} >= w);
                    else
                        idx = 1:size(trXX,2);
                    end
                    
                    if parallel
                        try
                            [~, ~, ~, ~, betas] = plsregress(trXX(:,idx), trYY, nc,'Options',statset('UseParallel',true));
                            err=false;
                        catch
                            err=true;
                        end
                    else
                        try
                            [~, ~, ~, ~, betas] = plsregress(trXX(:,idx), trYY, nc);
                            err=false;
                        catch
                            err=true;
                        end
                    end
                    
                    if ~err
                        y_pred = [ones(size(teXX(:,idx), 1), 1) teXX(:,idx)] * betas;
                        mse = mean(sum((teYY - y_pred).^2, 2));
                        correlation = corr(teYY, y_pred,'rows','complete');
                        correlation = mean(diag(correlation),'omitnan');
                        custom_loss = alpha * mse - (1 - alpha) * correlation;
                        custom_loss_sum = custom_loss_sum + custom_loss;
                    end
                end
                custom_loss_avg = custom_loss_sum / cvi{r}{i}.NumTestSets;
                if custom_loss_avg < bestCustomLoss && custom_loss_avg ~= 0
                    %disp(num2str(custom_loss_avg))
                    bestCustomLoss = custom_loss_avg;
                    bestNumComponents = nc;
                    bestSS = w;
                end
            end
        end
        winC(r,i) = bestNumComponents;
        winCLoss(r,i) = bestCustomLoss;
        winW(r,i) = bestSS;

        if standardizeX
            [trX,C,S] = normalize(trX);
            teX = normalize(teX,'Center',C,'Scale',S);
        end
        if standardizeY
            [trY,C2,S2] = normalize(trY);
            teY = normalize(teY,'Center',C2,'Scale',S2);
        end

        % Perform PLS regression with the best number of components
        if parallel
            if doSS
                [fk{r,i},~,~,~,~,~,~,~,~,~,~,~,~,~] = stabSel(trX,trY,...
                    'parallel',false,'stnd',false,'compPars',true,'samType','bootstrap','filter',false,'filterThresh',1,...
                    'maxVars',[],'thresh',bestSS,'fdr',[],'numFalsePos',[],'nComp',[min(ncrtmp):1:max(ncrtmp)],'selAlgo',...
                    'pls','rep',80,'prop',0.5,'parallel',true,'verbose',false); %maxVars 50 gives 16
                
                try
                    [~, ~, ~, ~, betas] = plsregress(trX(:,fk{r,i}), trY, bestNumComponents,'Options',statset('UseParallel',true));
                catch
                    [~, ~, ~, ~, betas] = plsregress(trX(:,fk{r,i}), trY, length(fk{r,i}));
                end
                
            else
                [~, ~, ~, ~, betas] = plsregress(trX, trY, bestNumComponents,'Options',statset('UseParallel',true));
            end
        else
            if doSS
                [fk{r,i},~,~,~,~,~,~,~,~,~,~,~,~,~] = stabSel(trX,trY,...
                    'parallel',false,'stnd',false,'compPars',true,'samType','bootstrap','filter',false,'filterThresh',1,...
                    'maxVars',[],'thresh',bestSS,'fdr',[],'numFalsePos',[],'nComp',[min(ncrtmp):1:max(ncrtmp)],'selAlgo',...
                    'pls','rep',80,'prop',0.5,'parallel',true,'verbose',false); %maxVars 50 gives 16
                try
                    [~, ~, ~, ~, betas] = plsregress(trX(:,fk{r,i}), trY, bestNumComponents);
                catch
                    [~, ~, ~, ~, betas] = plsregress(trX(:,fk{r,i}), trY, length(fk{r,i}));
                end
            else
                [~, ~, ~, ~, betas] = plsregress(trX, trY, bestNumComponents);
            end
        end

        % Predict the response values for the test data using the PLS model
        if doSS
            pls_yh(cvo{r}.test(i), :,r) = [ones(size(teX(:,fk{r,i}), 1), 1) teX(:,fk{r,i})] * betas;
        else
            pls_yh(cvo{r}.test(i), :,r) = [ones(size(teX, 1), 1) teX] * betas;
        end
        yNorm(cvo{r}.test(i), :,r) = teY;
        if saveBetas
            allBetas{r}{i} = betas;
        end
    end
    if tmpsave
       save('temporary_pls2.mat','-v7.3') 
    end
end
