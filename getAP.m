function [ap, mae, mase, baseM, mscale] = getAP(y,ypred,c,trimZero)
% This function will compute accuracy percent. Given true values in y,
% predicted y values in ypred (1-D vectors), and a cvpartition object in c,
% getAP will loop through your test folds and get mean absolute error
% (mae), then scale it to a baseline model that guesses using the mean of
% the training data (mase). Finally, getAP will express mase in percent
% units (ap). Note, output baseM is just the mae for the baseline model.
% The input argument trimZero will clip predicted values below 0 to be 0. 

if isempty(trimZero)
    trimZero = False;
end
id = find(y< 0);
if trimZero && ~isempty(id)
    trimZero = False;
    warning('you have negative values in y, it makes no sense to clip negative predictions')
end

if trimZero
    id = find(ypred < 0);
    ypred(id) = 0;
end

for i = 1:c.NumTestSets
    mscale(test(c,i)) = mean(y(training(c,i)));
end

mae = mean(abs(ypred-y));
baseM = mean(abs(mscale'-y));
mase = mae/baseM;
ap = 100*(1-mase);
