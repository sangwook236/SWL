function [model, loglikHist] = mixVonMisesFit_sangwook(data, nmix, varargin)
%% Fit a mixture of von Mises distributions via MLE/MAP (using EM)
%
%
%% Inputs
%
% data     - data(i, :) is the ith case, i.e. data is of size n-by-d
% nmix     - the number of mixture components to use
%
% This file is from pmtk3.googlecode.com

% for von Mises distribution
%	[ref] ${PMTK_HOME}/toolbox/LatentVariableModels/mixGauss/mixGaussFit.m

[initParams, prior, mixPrior, EMargs] = ...
    process_options(varargin, ...
    'initParams'        , [], ...
    'prior'             , [], ...
    'mixPrior'          , []);
[n, d]      = size(data);
model.type  = 'mixVonMises';
model.nmix  = nmix;
model.d     = d;
model       = setMixPrior(model, mixPrior);

initFn = @(m, X, r)initVonMises_sangwook(m, X, r, initParams, prior); 
[model, loglikHist] = emAlgo(model, data, initFn, @estep, @mstep , ...
                            'verbose', true, EMargs{:});
end

function model = initVonMises_sangwook(model, X, restartNum, initParams, prior)
%% Initialize 
nmix = model.nmix; 
if restartNum == 1
    if ~isempty(initParams)
        mu              = initParams.mu;
        kappa           = initParams.kappa;
        model.mixWeight = initParams.mixWeight;
    else
        %[mu, Sigma, model.mixWeight] = kmeansInitMixGauss(X, nmix);

        %[cid, alpha, mu] = circ_clust(X', nmix);  % out-of-memory
        %num_samples = size(X, 1);
        %mu = mu';
        %for kk = 1:nmix
        %	kappa(:,kk) = circ_kappa(X(cid == kk));
        %	model.mixWeight(:,kk) = sum(cid == kk) / num_samples;
        %end;

        [mu, kappa, model.mixWeight, step] = em_MovM(X', nmix, [], [], [], 100, 1e-2);
        num_samples = size(X, 1);
    end
else
    mu              = randn(d, nmix);
    regularizer     = 2; 
    %Sigma           = stackedRandpd(d, nmix, regularizer); 
    % TODO [check] >>
    kappa           = rand(d, nmix) + regularizer;
    model.mixWeight = normalize(rand(1, nmix) + regularizer); 
end
%model.cpd = condGaussCpdCreate(mu, Sigma, 'prior', prior);
model.cpd = condVonMisesCpdCreate_sangwook(mu, kappa, 'prior', prior);
end


function [ess, loglik] = estep(model, data)
%% Compute the expected sufficient statistics
[weights, ll] = mixVonMisesInferLatent_sangwook(model, data); 
cpd           = model.cpd;
ess           = cpd.essFn(cpd, data, weights); 
ess.weights   = weights; % useful for plottings
loglik        = sum(ll) + cpd.logPriorFn(cpd) + model.mixPriorFn(model); 
end

function model = mstep(model, ess)
%% Maximize
cpd             = model.cpd;
model.cpd       = cpd.fitFnEss(cpd, ess); 
model.mixWeight = normalize(ess.wsum + model.mixPrior - 1); 
end

%

