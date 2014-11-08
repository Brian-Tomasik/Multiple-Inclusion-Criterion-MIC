% README for MIC code
% 16 June 2009
% 
% This file contains all the functions needed to run stepwise regression
% using the three coding schemes described in 
%  Paramveer S. Dhillon, Brian Tomasik, Dean Foster, and Lyle Ungar.
%   "Multi-Task Feature Selection using the Multiple Inclusion Criterion
%   (MIC)." ECML PKDD '09.
% Given an n x m matrix X of features and an n x h matrix Y of responses,
% the algorithms can be run as follows:
%  mic(X,Y,'method','Partial MIC')
%  mic(X,Y,'method','Full MIC')
%  mic(X,Y,'method','RIC')
% For example:
%  X = randn(100,5);
%  w = zeros(5,3); w(2,2) = 1; w(4,1) = -1;
%  Y = X * w + randn(100,3) * 0.1;
%  mic(X,Y,'method','Partial MIC');
% Several functions have additional parameters documented below; however,
% they were all set to the current default settings for the experiments
% in the above-cited paper.
% The amount of output can be increased by changing the 'verbosity' setting
% from 0 to 1 or 2.

function [betaHat YHat] = mic(X, Y, varargin)
% Many of the parameters are self-explanatory, but here are some that
% need elaboration:
%  method: 'Partial MIC' uses the log^* k + c_h penalty to code k, the
%   number of nonzero responses. To use log h to code k instead, set
%   method to 'partial_logh'. The 'Full MIC' setting is what you'd expect.
%   'indep', however, is NOT 'RIC' from the paper: Rather, it uses
%   the regular MIC likelihood term, just with a diagonal sigmaHat, and
%   employs h-1 semicolons to distinguish which features go with which
%   response. To run RIC as described in the paper, use
%   'RIC', which calls mic on each response
%   separately, so that (with h==1) the result will be equivalent to RIC.
%  hypothesis: Default is 'no'. Otherwise, do the hypothesis version of MIC,
%   in which we regress on each feature separately.
%  eyeParam: sigmaHat is a convex combination of the full cov estimate and
%   a diagonal one. sigmaHat = (1-eyeParam)*(full estimate) + eyeParam *
%   (diagonal estimate). The default is eyeParam==1 because, by default, we
%   want sigmaHat to be diagonal.
%  forceDiagSigmaHat: Pretty much redundant -- this just makes sure that
%   eyeParam is set to 1 and that we're regularizing sigmaHat using the
%   convex-combination approach.
%  useAltSigmaHat: Default is false. If true, recompute sigmaHat each time
%   we try adding a new feature. Tends to cause overfitting....
%  shrinkMethod: Different ways to regularize sigmaHat. 'givenEyeParam' is
%   the convex-combination method described above and seems to work best.
%   To see the details of others, look in the documentation for the
%   sampleCov function.
%  costPerCoeff: Actually set to 2, though in a roundabout way that first
%   sets it to DEFAULT (== -3) and then changes it. This is to allow it to
%   differ for the hypothesis and non-hypothesis versions.
%  nBestFeaturesCheck: During an iteration of stepwise search, look through
%   only this number of the best features, where 'best' means 'biggest
%   improvement in log-likelihood when all responses are added'.
%  altHypAndDiagSigmaHat: As a warning says below, don't use this: It
%   causes overfitting.
%  pForFeatureCost: Will be set to p unless another value is provided. It's
%   not meant to be changed by an external user, just when we call
%   ourselves in the hypothesis-code for-loop.

% Define constants.
NOT_PROVIDED = -1;
NONE = -2;
DEFAULT = -3;

% Parse input.
parser = inputParser;
parser.FunctionName = 'mic';
parser.addRequired('X', @(x)or(isnumeric(x),islogical(x)));
parser.addRequired('Y', @(x)or(isnumeric(x),islogical(x)));
parser.addParamValue('method', 'Partial MIC', ...
    @(x)any(strcmpi(x, {'partial_logh','Partial MIC','Full MIC',...
    'indep','RIC'})));
% WARNING: 'indep' is NOT 'Independent' as used in the ICML paper; rather
% 'RIC' is. See the explanation in the parameter-definition
% section at the beginning of the file.
parser.addParamValue('hypothesis', 'no', @(x)any(strcmpi(x, {'no','bonferroni-style', 'simes-style'})));
parser.addParamValue('verbosity', 0, @(x)or(x==0,or(x==1,x==2)));
parser.addParamValue('eyeParam', 1, @(x)and(isnumeric(x),x>=0)); % diagonal!!
parser.addParamValue('forceDiagSigmaHat', 1, @(x)or(x==true,x==false));
parser.addParamValue('useAltSigmaHat',0,@(x)or(x==0,x==1));
parser.addParamValue('shrinkMethod', 'givenEyeParam', ...
    @(x)any(strcmpi(x, {'givenEyeParam','LedoitWolf', 'SchaferEtAl'})));
parser.addParamValue('costPerCoeff', DEFAULT);
parser.addParamValue('omitInterceptBetaHats', 0, @(x)or(x==0,x==1));
% Use the above for hypothesis versions where we care about which
% features are chosen and don't want to deal with intercept.
parser.addParamValue('nBestFeaturesCheck',75,@(x)x>0); % set to Inf for maximum search, but slow
parser.addParamValue('altHypAndDiagSigmaHat', 0, @(x)or(x==0,x==1));
parser.addParamValue('pForFeatureCost', NOT_PROVIDED)
parser.parse(X, Y, varargin{:});
X = parser.Results.X;
Y = parser.Results.Y;
method = parser.Results.method;
hypothesis = parser.Results.hypothesis; % Do hypothesis-testing version?
verbosity = parser.Results.verbosity;
eyeParam = parser.Results.eyeParam;
forceDiagSigmaHat = parser.Results.forceDiagSigmaHat;
useAltSigmaHat = parser.Results.useAltSigmaHat;
shrinkMethod = parser.Results.shrinkMethod;
costPerCoeff = parser.Results.costPerCoeff;
omitInterceptBetaHats = parser.Results.omitInterceptBetaHats;
nBestFeaturesCheck = parser.Results.nBestFeaturesCheck;
altHypAndDiagSigmaHat = parser.Results.altHypAndDiagSigmaHat;
pForFeatureCost = parser.Results.pForFeatureCost;

% WARNING: Don't use this option because it causes overfitting!!
altHypAndDiagSigmaHat = false;

% Do hypothesis MIC?
if(strcmpi(hypothesis,'no')) % Ordinary stepwise MIC
    % Set costPerCoeff for non-hypothesis version.
    if(costPerCoeff==DEFAULT)
        costPerCoeff = 2;
    end

    % Require that exactly the first column of X be 1s.
    firstColOnes = all(X(:,1)==1);
    if(~firstColOnes)
        X = [ones(size(X,1),1) X];
        if(omitInterceptBetaHats)
            warning('Adding a column of 1s to X. It will not show up in betaHat, so Yhat ~= X * betaHat.');
        else
            warning('Adding a column of 1s to X. This will be reflected in the dimensions of betaHat.');
        end
    end
    assert(~any(all(X(:,2:end)==1)),'No other column should be all 1s.');

    % Initialize values.
    [n,p] = size(X);
    if(pForFeatureCost==NOT_PROVIDED)
        pForFeatureCost = p;
    end
    [n2,h] = size(Y);
    assert(n == n2);
    if(forceDiagSigmaHat || ismember(method,{'indep'}))
        shrinkMethod = 'givenEyeParam';
        eyeParam = 1;
    end

    % Create a 'model' struct to store important state.
    model.X = X;
    model.Y = Y;
    model.h = h;
    model.verbosity = verbosity;
    model.useAltSigmaHat = useAltSigmaHat;
    model.altHypAndDiagSigmaHat = altHypAndDiagSigmaHat;
    model.q = ones(1,h); % # features for each response

    % Store a method that, given residuals, computes sigmaHat.
    model.isCorrMatrix = false; % change to 'true' if we ever standardize X and Y
    model.sigmaHatOfResiduals = ...
        @(residuals)sampleCov(residuals,'eyeParam',eyeParam,...
        'shrinkMethod',shrinkMethod,...
        'detMakePosStruct',struct('do',1,'isCorrMatrix',model.isCorrMatrix));
    model.sigmaHatIsDiag = and(strcmpi(shrinkMethod,'givenEyeParam'),...
        eyeParam>.999);

    % Start computing betaHat.
    model.betaHat = zeros(p,h);
    model.betaHat(1,:) = mean(model.Y); % Initial intercept terms.
    model.curResiduals = model.Y - model.X * model.betaHat;
    model.curNullSigmaHat = model.sigmaHatOfResiduals(model.curResiduals);
    model.curNullSigmaHatInv = inv(model.curNullSigmaHat);
    if(verbosity>1)
        fprintf('Initial sigmaHat:\n');
        initialSigmaHat = model.curNullSigmaHat;
        initialSigmaHat = initialSigmaHat
    end

    % Split between RIC and the other two methods.
    if(strcmpi(method,'RIC'))
        % Ignore everything we've done so far, and get betaHat by doing h
        % separate calls to MIC indep with h==1.
        for iResponse = 1:h
            model.betaHat(:,iResponse) = mic(X,Y(:,iResponse),...
                'method','indep',...
                'hypothesis',hypothesis,...
                'costPerCoeff',costPerCoeff-1,... % This is important. We reduce cost per coefficient because 'indep' normally pays 1 bit too many because of semicolons.
                'omitInterceptBetaHats',0,... % Not now. Do that at the end of ourselves.
                'verbosity',verbosity,...
                'eyeParam',eyeParam,...
                'forceDiagSigmaHat',forceDiagSigmaHat,...
                'useAltSigmaHat',useAltSigmaHat,...
                'shrinkMethod',shrinkMethod,...
                'altHypAndDiagSigmaHat',altHypAndDiagSigmaHat,...
                'nBestFeaturesCheck',nBestFeaturesCheck,...
                'pForFeatureCost',pForFeatureCost);
        end
    else
        % Store model coding costs for later use.
        if(strcmpi(method,'Full MIC'))
            % We should only ever be looking at costs for the Full MIC model.
            model.modelCostOfK = repmat(NaN,1,h);
            model.modelCostOfK(h) = log2(pForFeatureCost) + ... % log2(pForFeatureCost) for the feature cost
                nonFeatureCodingPenalty(method,h,h,costPerCoeff);
        else
            model.modelCostOfK = log2(pForFeatureCost) + ... % log2(pForFeatureCost) for the feature cost
                arrayfun(@(k)nonFeatureCodingPenalty(method,h,k,costPerCoeff),1:h);
        end

        % Initialize coding cost.
        if(ismember(method,{'indep'}))
            model.codingCost = h-1; % code the h-1 semicolons
        else
            model.codingCost = 0;
        end

        % Store info on what features we've used.
        model.nFeatures = 1;
        model.featuresNotYetUsed = 2:p;

        % Below is model with just intercept terms. This assumes no cost to code the
        % intercept coefficients (cheating...).
        [model.likelihoodCost shouldEqualBetaHat] = ...
            likelihoodCostOfResponseSubset(ones(1,h),1,model);
        assert(all(all(shouldEqualBetaHat-model.betaHat<.00001)));

        if(verbosity>0)
            fprintf('\n\n');
            fprintf('Method=%s, costPerCoeff=%i\n',method,costPerCoeff);
            print_diagnostics(model,1,h,NaN,NaN);
        end

        % Store previous values of the costs for use in printing out changes
        % later on.
        prevCodingCost = model.codingCost;
        prevLikelihoodCost = model.likelihoodCost;
        bestCostSoFar = model.codingCost + model.likelihoodCost;

        % Try adding features until it no longer helps.
        for nFeatures = 2:p
            % Find the next best feature.
            bestFeatureToAdd = NONE;

            % Choose a set of features to examine to see if we want to add
            % them. If we don't care about speed, look through all the
            % remaining features. Otherwise, choose a subset of them.
            
            % Sort the unused features by log-likelihood of the Full MIC model
            % and, if we're optimizing, check the smallest ones only.
            negLogLikeCosts = arrayfun(@(feature)likelihoodCostOfResponseSubset(ones(1,h),feature,model),model.featuresNotYetUsed);
            [sortedNegLogLikeCosts indNegLogLikeCosts]=sort(negLogLikeCosts);
            checkTheseFeatures = model.featuresNotYetUsed(indNegLogLikeCosts(1:min(nBestFeaturesCheck,length(indNegLogLikeCosts))));
            % TODO: Sort checkTheseFeatures in increasing order by cost at
            % that index.
            % TODO: sort the responses and do similar branch+bound there!
            % TODO: improve my lower bound on k in the response
            % branch+bound to be the one in the paper.

            % Now check the features.
            if(verbosity>2 && ~strcmpi(method,'Full MIC'))
                fprintf('\tConsidering feature');
            end
            for curFeature = checkTheseFeatures
                % TODO: Add branch+bound check to see if even bother with
                % this feature. Is the lower bound on its description
                % length (i.e., its residual cost plus lg p) smaller than
                % the best actual description length seen so far?
                if(verbosity>2 && ~strcmpi(method,'Full MIC'))
                    fprintf(' %i,',curFeature);
                end

                % Get costs if add.
                if(strcmpi(method,'Full MIC'))
                    [curNegLogLike candidateNewBetaHat] = ...
                        likelihoodCostOfResponseSubset(ones(1,h),curFeature,model);
                    costIfAdded = curNegLogLike + ...
                        model.codingCost + ...
                        model.modelCostOfK(h);
                else
                    [candidateNewBetaHat costIfAdded curNegLogLike] = ...
                        bestResponseSubset(curFeature,model,bestCostSoFar);
                end

                % Is it low enough?
                if(costIfAdded < bestCostSoFar)
                    bestFeatureToAdd = curFeature;
                    bestCostSoFar = costIfAdded;
                    bestLikelihoodCost = curNegLogLike;
                    bestNewBetaHat = candidateNewBetaHat;
                    if(verbosity>2 && ~strcmpi(method,'Full MIC'))
                        fprintf('\n\t\tNew cost = %f with %i responses for feature %i.\n\tconsidering feature',bestCostSoFar,nnz(bestNewBetaHat(curFeature,:)),curFeature);
                    end
                end
            end

            % Add that feature to the model.
            if(bestFeatureToAdd~=NONE)
                % Update state.
                model.nFeatures = model.nFeatures + 1;
                model.likelihoodCost = bestLikelihoodCost;
                model.betaHat = bestNewBetaHat;
                cur_k = nnz(model.betaHat(bestFeatureToAdd,:));
                model.codingCost = model.codingCost + model.modelCostOfK(cur_k);
                model.curResiduals = model.Y - model.X * model.betaHat;
                model.curNullSigmaHat = model.sigmaHatOfResiduals(model.curResiduals);
                model.curNullSigmaHatInv = inv(model.curNullSigmaHat);
                model.q = arrayfun(@(col)nnz(model.betaHat(:,col)),1:model.h);

                % Remove the feature from list of unused features.
                indexOfFeatureToRemove = ...
                    find(model.featuresNotYetUsed==bestFeatureToAdd);
                assert(length(indexOfFeatureToRemove)==1);
                model.featuresNotYetUsed(indexOfFeatureToRemove) = [];

                % Print diagnostics.
                print_diagnostics(model,bestFeatureToAdd,cur_k,prevLikelihoodCost,prevCodingCost);
                if(verbosity>1)
                    curSigmaHat = model.curNullSigmaHat;
                    curSigmaHat = curSigmaHat
                end

                % Store previous values of the costs for use in printing out changes
                % later on.
                prevCodingCost = model.codingCost;
                prevLikelihoodCost = model.likelihoodCost;
                assert(abs(bestCostSoFar-(model.codingCost+model.likelihoodCost))<.001);
            else
                break;
            end
        end
    end % End of split between 'RIC' and the other two methods.

    % Return betaHat.
    betaHat = model.betaHat;

    % Prepare return results.
    YHat = X * betaHat;
    if(omitInterceptBetaHats)
        betaHat = betaHat(2:end,:);
    end
    if(verbosity>1)
        fprintf('Final sigmaHat:\n');
        finalSigmaHat = model.curNullSigmaHat;
        finalSigmaHat = finalSigmaHat
    end
    if(verbosity>0)
        min_p_and_20 = min(size(betaHat,1),20);
        fprintf('First %i rows of betaHat:\n',min_p_and_20);
        firstRowsOfBetaHat = betaHat(1:min_p_and_20,:)
    end
elseif(ismember(hypothesis,{'bonferroni-style','simes-style'}))
    % Gives FDR's around 0.05-0.1.
    if(costPerCoeff==DEFAULT)
        costPerCoeff = 2;
    end

    % Should not have a column of 1's in X.
    assert(~any(all(X==1)),'No column of X should be all 1s.');

    % Initialize values.
    [n,p] = size(X);
    if(pForFeatureCost==NOT_PROVIDED)
        pForFeatureCost = p;
    end
    [n2,h] = size(Y);
    assert(n == n2);

    % Initialize output matrices.
    betaHat = zeros(p,h);
    YHat = repmat(NaN,n,h); % There is no single YHat for hypothesis version.

    % BONFERRONI-STYLE HYPOTHESIS
    if(strcmpi(hypothesis,'bonferroni-style'))
        % Fill in each row of betaHat as the result of regressing on just that
        % x.
        for iFeature = 1:p
            fprintf('Doing feature %i.\n',iFeature);
            cur_X = [ones(size(X,1),1) X(:,iFeature)];
            betaHat(iFeature,:) = mic(cur_X,Y,...
                'omitInterceptBetaHats',1,...
                'hypothesis','no',...
                'method',method,...
                'costPerCoeff',costPerCoeff,...
                'verbosity',0,...
                'eyeParam',eyeParam,...
                'forceDiagSigmaHat',forceDiagSigmaHat,...
                'useAltSigmaHat',useAltSigmaHat,...
                'shrinkMethod',shrinkMethod,...
                'altHypAndDiagSigmaHat',altHypAndDiagSigmaHat,...
                'nBestFeaturesCheck',nBestFeaturesCheck,...
                'pForFeatureCost',pForFeatureCost);
            nnz_cur_feature = nnz(betaHat(iFeature,:));
            if(nnz_cur_feature > 0)
                fprintf('\tPut in %i responses.\n',nnz_cur_feature);
            end
        end
    % SIMES-STYLE HYPOTHESIS
    elseif(strcmpi(hypothesis,'simes-style'))
        % Guess the number of features that will be put in. When we guess
        % right (which must happen eventually), that's our answer.
        guessed_n_features = 1;
        while(true)
            total_feature_cost = codeForIntegers(guessed_n_features, p) + ...
                log2nChoosek(p,guessed_n_features);
            amortized_feature_cost = total_feature_cost / guessed_n_features;
            p_such_that_lg_p_is_cost = 2^amortized_feature_cost;
            if(verbosity > 0)
                fprintf('Guessing %i features with amortized cost %.3f.\n',...
                    guessed_n_features,amortized_feature_cost);
            end
            betaHat = mic(X,Y,...
                'method',method,...
                'hypothesis','bonferroni-style',...
                'costPerCoeff',costPerCoeff,...
                'verbosity',verbosity,...
                'eyeParam',eyeParam,...
                'forceDiagSigmaHat',forceDiagSigmaHat,...
                'useAltSigmaHat',useAltSigmaHat,...
                'shrinkMethod',shrinkMethod,...
                'altHypAndDiagSigmaHat',altHypAndDiagSigmaHat,...
                'nBestFeaturesCheck',nBestFeaturesCheck,...
                'pForFeatureCost',p_such_that_lg_p_is_cost);
            n_features_betaHat = nnz(sum(abs(betaHat),2));
            if(verbosity > 0)
                fprintf('Actually %i features.\n\n',n_features_betaHat);
            end            
            if(n_features_betaHat==guessed_n_features)
                break; % We're done; we have the betaHat we want.
            elseif(n_features_betaHat<guessed_n_features)
                if(and(n_features_betaHat==0,guessed_n_features==1))
                    break; % That's fine; it just means betaHat has all 0s....
                else
                    error('n_features_betaHat fell below guessed_n_features. Should not happen.');
                end
            else
                guessed_n_features = guessed_n_features + 1;
            end
        end
    end
end

function sigmaHat = sampleCov(epsHat, varargin)
% function sigmaHat = sampleCov(epsHat, PARAM1, val1, PARAM2, val2, ...)
%  Compute a sample covariance matrix. Allows the option of a shrinkage
%   estimate.
%  Input:
%   epsHat: n x h matrix of the sample values of Y - YHat.
%   Optional parameter-value pairs:
%       'eyeParam'  The coefficient of "shrinkage" toward the
%              identity matrix. Default is 0.
%       'shrinkMethod'  One of the following:
%          'givenEyeParam'  Take a convex combination of the empirical
%              estimate and the identity with eyeParam in front of the
%              identity part.
%          'LedoitWolf' Use the shrunken estimator
%               recommended in the following paper:
%               Ledoit, O. and M. Wolf (2004) A Well-Conditioned Estimator for
%               Large-Dimensional Covariance Matrices. Journal of Multivariate
%               Analysis 88, 365-411.
%          'SchaferEtAl' Use the shrunken estimator based on
%               http://www.strimmerlab.org/software/corpcor/
%       'subtractEpsMean'   Should epsHat be forced to have sample
%               mean of zero by subtracting off its mean?
%               Default is true because Matlab's own 'cov' function
%               does this.
%       'df'    Degrees of freedom to subtract in the denominator
%               (e.g., possibly n-p for a regression with n data points
%               and p predictors). Default is 1 if subtractEpsMean is
%               true, else 0.
%   This function gives the same answer as Matlab's 'cov' function when
%   none of these parameters is set away from their default values.
%  Output:
%   sigmaHat: h x h sample covariance matrix.
%  Sample Call:
%   sampleCov([3 4; 2 1; 8 9; 2 2],'eyeParam',0.5,'df',2)

% Set defaults.
NOT_PROVIDED = -4;

% Get input.
parser = inputParser;
parser.FunctionName = 'sampleCov';
parser.addRequired('epsHat', @isnumeric);
parser.addParamValue('eyeParam', 0, @(x)and(isnumeric(x),x>=0));
parser.addParamValue('shrinkMethod', 'givenEyeParam', ...
    @(x)any(strcmpi(x, {'givenEyeParam','LedoitWolf', 'SchaferEtAl'})));
parser.addParamValue('subtractEpsMean', 1, @(x)or(x==true,x==false));
parser.addParamValue('detMakePosStruct', struct('do',0,'isCorrMatrix',1));
% This parameter holds a struct whose first field tells whether to
% make the resulting sigmaHat have positive determinant and whose second
% field tells whether, in the process, we should assume that sigmaHat
% represents a correlation matrix.
parser.addParamValue('df', NOT_PROVIDED, @(x)and(isnumeric(x),or(x>=0,x==NOT_PROVIDED)));
% If you want the best unbiased estimate, take df==# free parameters,
% including the intercept to the regression model.
parser.parse(epsHat, varargin{:});
epsHat = parser.Results.epsHat;
eyeParam = parser.Results.eyeParam;
subtractEpsMean = parser.Results.subtractEpsMean;
shrinkMethod = parser.Results.shrinkMethod;
detMakePosStruct = parser.Results.detMakePosStruct;
df = parser.Results.df;

% Set df if not given.
if(df==NOT_PROVIDED)
   if(subtractEpsMean)
       df = 1;
   else
       df = 0;
   end
end

% Get sizes.
[n, h] = size(epsHat);

% Subtract mean? Matlab's 'cov' function does.
if(subtractEpsMean)
    epsHat = epsHat - repmat(mean(epsHat),n,1);
end

% Check df is okay.
if df >= n
    error('df > n; using df = %i',df);
end

S_h = epsHat' * epsHat / (n-df);

% Choose method.
switch shrinkMethod
    case 'givenEyeParam'
        if(df==1)
            diagSigmaHat = diag(var(epsHat)); % divide by n-1
        else
            diagSigmaHat = diag(var(epsHat,1)); % divide by n
        end
        sigmaHat = (1-eyeParam) * S_h + eyeParam * diagSigmaHat; % Can change target.
        % Note: Above we're shrinking toward a diagonal matrix with the
        % *MLE* variances (divide by n, not n-1). This is because a t-test
        % implicitly uses this type of covariance matrix (because it's a
        % GLRT).
    case 'LedoitWolf'
        % Note: This tends to be pretty conservative in the amount it
        % shrinks. For instance, on one data set, the resulting sample
        % covariance matrices had determinant values on the order of those
        % obtained by taking eyeParam ~= 0.01.
        m_h = trace(S_h) / h;
        d_hSq = frobeniusNorm(S_h-m_h*eye(h));
        temp = 0;
        for i = 1:h
            temp = temp + frobeniusNorm(S_h(:,i) * S_h(:,i)' - S_h);
        end
        bBar_hSq = temp / h^2;
        b_hSq = min(bBar_hSq, d_hSq);
        a_hSq = d_hSq - b_hSq;
        sigmaHat = (a_hSq / d_hSq) * S_h + (b_hSq / d_hSq) * m_h * eye(h);
    case 'SchaferEtAl'
        % This case took much debugging, but it finally works! I checked it
        % against the real version of the function in R on two separate
        % examples, and both produced correct output. The second example
        % involved lambdaStar and lambda both between 0 and 1, so I didn't
        % just get lucky by having one of them get set to 1, which might
        % have hidden errors.
        
        % Compute rij and hat variances of rij.
        % See p. 28 of
        % http://strimmerlab.org/publications/shrinkcov2005.pdf
        X = epsHat;
        centeredX = X - repmat(mean(X),n,1);
        scaledCenteredX = centeredX ./ repmat(std(X),n,1); 
        % Standardize X first.
        
        R = zeros(h,h);
        VarHatR = zeros(h,h);
        for i = 1:h
            for j = 1:h
                wij = scaledCenteredX(:,i) .* scaledCenteredX(:,j); % wij is n x 1.
                wBarij = mean(wij);
                R(i,j) = (n / (n - df)) * wBarij;
                VarHatR(i,j) = (n / (n-df)^3) * sum((wij - wBarij) .^ 2);
            end
        end
        
        % Now compute lambdaStar.
        numerator = sum(sum(VarHatR)) - sum(diag(VarHatR));
        RSq = R .^ 2;
        denominator = sum(sum(RSq)) - sum(diag(RSq));
        if denominator==0
            lambdaStar = 1;
        else
            lambdaStar = numerator / denominator;
        end
        if(lambdaStar>1) lambdaStar = 1; end
        
        % Compute RStar.
        RStar = R * min(1,max(0,1-lambdaStar));
        for i = 1:h
            RStar(i,i) = 1;
        end
        
        % Look at sample variances.
        sigmaHat = zeros(h,h);
        variances = diag(S_h); % Get variances.
        target = median(variances);
        
        % Compute lambda.
        centerSq = centeredX .^ 2;
        numerator2 = sum((n^2 / (n-df)^3) * var(centerSq,1));
        % use the divide-by-n variances with var(.,1)
        denominator2 = sum((sum(centerSq) / (n-df) - target) .^ 2);
        if denominator2==0
            lambda = 1;
        else
            lambda = numerator2 / denominator2;
        end
        if(lambda > 1) lambda = 1; end
        
        % Compute sigmaHat (what the paper calls S*).
        newVariances = lambda * target + (1-lambda) * variances;
        newSDs = sqrt(newVariances);
        
        for i = 1:h
            for j = 1:h
                sigmaHat(i,j) = RStar(i,j) * newSDs(i) * newSDs(j);
            end
        end
    otherwise
        error('shrinkMethod does not match possibilities.');
end

% Check that determinant is positive?
if(detMakePosStruct.do)
   sigmaHat = detMakePos(sigmaHat,detMakePosStruct.isCorrMatrix);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = frobeniusNorm(A)
% function val = frobeniusNorm(A)
%  Input:
%   A: a x b matrix
val = trace(A*A') / size(A,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [M detM] = detMakePos(M,isCorrMatrix)
% Given a square matrix M, check that its determinant is positive. If not,
% make it positive definite using nearPD. Note that mathematically, 
% checking the determinant is necessary but not sufficient for positive 
% definiteness. However, checking, say, eigenvalues could get expensive,
% so we're lazy and don't bother -- if it's a problem, it will show up
% elsewhere.
% The 'isCorrMatrix' input argument is used by nearPD to decide whether
% the given matrix is a correlation matrix or a regular covariance matrix.

% Parse input.
parser = inputParser;
parser.FunctionName = 'detMakePos';
parser.addRequired('M', @(M)and(isnumeric(M),size(M,1)==size(M,2)));
parser.addRequired('isCorrMatrix', @(x)or(x==0,x==1));
parser.parse(M, isCorrMatrix);
M = parser.Results.M;
isCorrMatrix = parser.Results.isCorrMatrix;

% Check determinant. If not positive, change M.
detM = det(M);
if(detM <= 0)
    M = nearPD(M,'isCorrMatrix',isCorrMatrix);
    detM = det(M);
    assert(detM > 0, 'nearPD failed to make the matrix positive-definite.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = nonFeatureCodingPenalty(method, h, k, costPerCoefficient)
% nonFeatureCodingPenalty
%  Input:
%   method: Which coding method to use.
%   h: # responses.
%   k: # nonzero responses in the subset.
%   costPerCoefficient: How many bits does it take to code a coefficient?
%  Output:
%   val: The cost of coding nonzero parameters (if any), exept for the
%     cost of identifying which features are in the model; that part
%     comes later in the mic.m code.

if k == 0
    val = 0;
else
    switch method
        case 'indep'
            val = k * (costPerCoefficient + 1); % the +1 is for semicolons
        case 'partial_logh'
            val = log2(h) + log2nChoosek(h, k) + k * costPerCoefficient;
        case 'Partial MIC'
            val = codeForIntegers(k,h) + ...
                log2nChoosek(h,k) + k * costPerCoefficient;
        case 'conservativePartial'
            val = k * twoLogCodeCost(h/k) + k * costPerCoefficient;
            % !!!CAN CHANGE THE ABOVE TO MAKE IT correspond to actual
            % distances.
        case 'Full MIC'
            assert(h == k, ['Number of rejected  ' ...
                'hypotheses should equal number of responses.'])
            val = k * costPerCoefficient;
        otherwise
            error('Unknown method');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = codeForIntegers(k,h,varargin)
% codeForIntegers
%  Input:
%   k: A positive integer to be encoded.
%   h: The upper bound on the range of possible integer values. That is,
%     our code is for the integers {1, 2, ..., h}.
%   startAtZero (optional): If true, code {0, 1, ..., h}.
%  Output:
%   The cost of coding the integer k.
%
% Based on A universal prior for integers and estimation by minimum
% description length, Rissanen 1983.

% Parse input.
parser = inputParser;
parser.FunctionName = 'codeForIntegers';
parser.addRequired('k', @(x)and(isscalar(x),x>=0));
parser.addRequired('h', @(x)and(isscalar(x),x>0));
parser.addParamValue('startAtZero',0,@(x)or(x==0,x==1));
parser.parse(k,h,varargin{:});
k = parser.Results.k;
h = parser.Results.h;
startAtZero = parser.Results.startAtZero;

% Check inputs
if(~startAtZero)
    assert(k>0,'Cannot code 0 unless startAtZero==true.');
end
assert(h >= k, 'h must be >= k.')

% Starting at zero?
if(startAtZero)
    val = codeForIntegers(k+1,h+1,'startAtZero',0);
else
    % c_omega = log2(2.865); We would use this if we had a code over the
    % entire integers, but we actually just have a code on 1:h.
    % So calculate a more appropriate normalization factor.
    if h==6715
        normalizingConst = 1.2155; % Memoized values -- a hack but it works.
    elseif h==1000
        normalizingConst = 1.1987;
    elseif h==Inf
        normalizingConst = log2(2.865064);
    else
        probabilitiesOneToH = arrayfun(@(k)2^(-log2Star(k)),1:h);
        normalizingConst = log2(sum(probabilitiesOneToH));
    end
    val = normalizingConst + log2Star(k);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function cost = twoLogCodeCost(dist)
% twoLogCodeCost
%  Input:
%   dist: An integer distance in {0, 1, 2, ...} to be coded. Hopefully it's
%     fairly small.
%  Output:
%   cost: The coding cost of that distance using a code that's roughly
%     2*log(dist). We make some minor modifications for various cases.
assert(dist>=0,'dist can not be negative.');
% Add check that dist is an integer.
if(dist==0)
    cost = 2*ceil(log2(dist+2));
else
    cost = 2*ceil(log2(dist+1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log2Star
%  Input:
%   x: A positive number.
%  Output:
%   val: log2*(x)
function val = log2Star(x)
assert(x>0,'Cannot take log2Star of a non-positive number.');
val = 0;
curTerm = log2(x);
while(curTerm > 0)
   val = val + curTerm;
   curTerm = log2(curTerm);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = log2nChoosek(n,k)
% log2nChosek
%  Input:
%   n: A nonnegative integer.
%   k: A nonnegative integer <= n.
%  Output:
%   log base 2 of n choose k, computed to take advantage of the fact that
%   log grows slowly, so that values can be found even for very large
%   n and k.

% Check inputs
assert(isscalar(n) && isscalar(k), 'n and k must be a scalar.')
%assert(???, 'n and k must be integers.');
% Is there a way to check that a number is an integer? isinteger doesn't
% work....
assert(n >= 0 && k >= 0, 'n and k must be positive.')
assert(n >= k, 'n must be >= k.')

% Compute, using a formula that works even for very large n and k.
val = sum(log2(n-k+1:n)) - sum(log2(1:k));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = negLogLikelihood(epsHat,sigmaHat,varargin)
% negLogLikelihood
%  Input:
%   epsHat: An n x h matrix of residuals for a particular subset of
%     features.
%   sigmaHat: The sample covariance matrix to use.
%   Optional parameters:
%    'sigmaHatInv': The inverse of sigmaHat.
%  Output:
%   out: The negative log likelihood term.

% Define constants.
CHANGE_ME = -5;
PRINT_WARNING = 0;

% Parse input.
parser = inputParser;
parser.FunctionName = 'negLogLikelihood';
parser.addRequired('epsHat', @isnumeric);
parser.addRequired('sigmaHat', @isnumeric);
parser.addParamValue('sigmaHatInv', CHANGE_ME, @isnumeric);
parser.addParamValue('isCorrMatrix', false, @(x)or(x==0,x==1));
parser.addParamValue('howMuchCheckPosDef', 2, @(x)or(x==1,x==2)); % 2 == all the time, 1 == only if det <= 0.
parser.addParamValue('sigmaHatIsDiag', 0, @(x)or(x==0,x==1));
parser.addParamValue('altHypAndDiagSigmaHat', 0, @(x)or(x==0,x==1));
parser.addParamValue('q', CHANGE_ME, @(x)or(all(x>=0),all(x==CHANGE_ME)));
% q is a 1 x h array of the number of features in model, including the intercept
% It's not needed unless altHypAndDiagSigmaHat==1.
parser.parse(epsHat, sigmaHat, varargin{:});
epsHat = parser.Results.epsHat;
sigmaHat = parser.Results.sigmaHat;
sigmaHatInv = parser.Results.sigmaHatInv;
isCorrMatrix = parser.Results.isCorrMatrix;
howMuchCheckPosDef = parser.Results.howMuchCheckPosDef;
sigmaHatIsDiag = parser.Results.sigmaHatIsDiag;
altHypAndDiagSigmaHat = parser.Results.altHypAndDiagSigmaHat;
q = parser.Results.q;

% Get n and h.
[n h] = size(epsHat);

if(~altHypAndDiagSigmaHat) % Do it the usual way.
    % If sigmaHat isn't positive-definite, recompute one that is.
    if(~sigmaHatIsDiag && howMuchCheckPosDef==2)
        if(any(eig(sigmaHat)<=0)) % Then sigmaHat isn't positive-definite.
            sigmaHat = nearPD(sigmaHat,'isCorrMatrix',isCorrMatrix);
            % A matrix being positive-definite implies that its inverse is.
            sigmaHatInv = inv(sigmaHat);
            fprintf(2,'\nChanging sigmaHat to be positive-definite.');
        end
    end

    % If necessary, compute sigmaHatInv.
    if(sigmaHatInv==CHANGE_ME) sigmaHatInv = inv(sigmaHat); end

    % Get input dimensions.
    [h1, h2] = size(sigmaHatInv);
    assert(h==h1 && h==h2,'sigmaHatInv not square!');
    [h3 h4] = size(sigmaHat);
    assert(and(h==h3,h==h4),'sigmaHat not same dimensions as sigmaHatInv.');

    % Check determinant of sigmaHat. If sigmaHatIsDiag==1, the determinant is
    % easy....
    if(~sigmaHatIsDiag)
        detSigmaHat = det(sigmaHat);
    else
        detSigmaHat = prod(diag(sigmaHat));
    end
    if(detSigmaHat <= 0 && howMuchCheckPosDef >= 1)
        out = negLogLikelihood(epsHat,nearPD(sigmaHat,'isCorrMatrix',isCorrMatrix),...
            'isCorrMatrix',isCorrMatrix,'howMuchCheckPosDef',...
            howMuchCheckPosDef);
    else
        % Compute.
        firstTerm = (n*h/2) * log2(2*pi) + (n/2)*log2(detSigmaHat);
        undividedSecondTerm = 0;
        % If sigmaHatInv is diagonal, use a shortcut in computing
        % this second term.
        %sigmaHatIsDiag = all(all((sigmaHatInv~=0)==eye(size(sigmaHatInv))));
        if(sigmaHatIsDiag)
            summedEpsHatSq = sum(epsHat .^ 2);
            undividedSecondTerm = summedEpsHatSq * diag(sigmaHatInv); % row * column
        else
            for i = 1:n
                undividedSecondTerm = undividedSecondTerm + ...
                    epsHat(i,:) * sigmaHatInv * epsHat(i,:)';
            end
        end
        if(undividedSecondTerm <= 0)
            fprintf(2,'sigmaHatInv was supposed to be positive definite!...\n\n');
        end
        out = firstTerm + undividedSecondTerm / (2 * log(2));
        
        % Check to see if out is negative. It would be unusual if it is, but
        % not necessarily an error, because we're dealing with density
        % functions and not discrete probabilities here.
        if(out <= 0)
            if(PRINT_WARNING > 0)
                fprintf(2,'-log2(likelihood) <= 0: %g.\n',out); % print to stderr
            end
        end
    end
else % case: altHypAndDiagSigmaHat
    dividedSumSqErr = sum(epsHat.^2) ./ (n-q);
    onePlusLogSumSq = 1 + log(2 * pi * dividedSumSqErr); % 1 x h vector
    out = (n/2) * sum(onePlusLogSumSq);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [negLogLike candidateNewBetaHat] = ...
    likelihoodCostOfResponseSubset(subset,curFeature,model,varargin)

% Parse input.
parser = inputParser;
parser.FunctionName = 'likelihoodCostOfResponseSubset';
parser.addRequired('subset');
parser.addRequired('curFeature');
parser.addRequired('model');
parser.parse(subset,curFeature,model,varargin{:});
subset = parser.Results.subset;
curFeature = parser.Results.curFeature;
model = parser.Results.model;

% Use likelihood of MLE betaHat values?
% Compute what the new betaHat would be if curFeature were to have
% nonzero coefficients with the subset 'subset' of responses.
% We don't want to set model's betaHat field to this new betaHat,
% though, because we may not end up using it, so we store it as a
% temporary 'candidateNewBetaHat'.
oldBetaHat = model.betaHat;
candidateNewBetaHat = oldBetaHat;
cur_q = model.q + subset;
for col = find(subset)
    if(oldBetaHat(curFeature,col)==0)
        featuresForThisResponse = fast_union(curFeature,find(oldBetaHat(:,col)));
        candidateNewBetaHat(featuresForThisResponse,col) = ...
            model.X(:,featuresForThisResponse) \ model.Y(:,col);
    end
end

% Get residuals.
candidateEpsHat = model.Y - model.X * candidateNewBetaHat;

if(model.altHypAndDiagSigmaHat)
    [n h] = size(candidateEpsHat);
    dividedSumSqErr = sum(candidateEpsHat.^2) ./ (n-cur_q);
    onePlusLogSumSq = 1 + log(2 * pi * dividedSumSqErr); % 1 x h vector
    negLogLike = (n/2) * sum(onePlusLogSumSq);
else
    % Recompute sigmaHat?
    if(model.useAltSigmaHat)
        sigmaHat = model.sigmaHatOfResiduals(candidateEpsHat);
        % This isn't the MLE for sigmaHat, but oh well!
        sigmaHatInv = inv(sigmaHat);
    else
        sigmaHat = model.curNullSigmaHat;
        sigmaHatInv = model.curNullSigmaHatInv;
    end

    % Compute -log2(likelihood).
    negLogLike = negLogLikelihood(candidateEpsHat,sigmaHat,...
        'sigmaHatInv',sigmaHatInv,'isCorrMatrix',model.isCorrMatrix,...
        'sigmaHatIsDiag',model.sigmaHatIsDiag,...
        'altHypAndDiagSigmaHat',model.altHypAndDiagSigmaHat);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out_list = fast_union(scalar, in_list)
% Because the function union is so slow (enough to be one of the two major
% bottlenecks of the MIC code!), I'm writing a faster version that handles
% this particular case: where the first argument is just a number.
if(any(scalar==in_list))
    out_list = in_list;
else
    [nRows nCols] = size(in_list);
    if(nCols > 1)
        out_list = [scalar, in_list]; % concatenate horizontally
    else
        out_list = [scalar; in_list]; % concatenate vertically
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [candidateNewBetaHat nonFeatureCostIfAdded bestNegLogLike] = ...
    bestResponseSubset(curFeature,model,lowestNonfeatureCostEver,varargin)

% Define constants.
NONE_CONST = -7;

% Parse input.
parser = inputParser;
parser.FunctionName = 'bestResponseSubset';
parser.addRequired('curFeature');
parser.addRequired('model');
parser.addRequired('lowestNonfeatureCostEver');
parser.addParamValue('stopEarly',0,@(x)or(x==0,x==1));
parser.parse(curFeature,model,lowestNonfeatureCostEver,varargin{:});
curFeature = parser.Results.curFeature;
model = parser.Results.model;
lowestNonfeatureCostEver = parser.Results.lowestNonfeatureCostEver;
stopEarly = parser.Results.stopEarly;

% Create a function of subsets giving their likelihood cost. A subset is
% a 1 x h vector like [0 0 1 0].
subsetNegLogLikeValue = @(subset)likelihoodCostOfResponseSubset(subset,curFeature,model);

% Get a lower bound on the model cost for each k > 0. If this lower bound
% isn't below lowestNonfeatureCostEver, don't even bother looking.
lowerBoundsEachK = subsetNegLogLikeValue(ones(1,model.h)) + ...
    model.codingCost + ...
    model.modelCostOfK;
maxKToCheck = find(lowerBoundsEachK<lowestNonfeatureCostEver,1,'last');

% Initialize values for subset search.
curSubset = zeros(1,model.h);
candidateNewBetaHat = NONE_CONST;

% Iterate over how many entries of the subset are nonzero.
for k = 1:maxKToCheck
    % Find best subset of size k.
    newNegLogLikes = repmat(Inf, 1, model.h);
    newCandidateBetaHats = cell(1,model.h);
    for i = find(curSubset==0)
        [newNegLogLikes(i) newCandidateBetaHats{i}] = ...
            subsetNegLogLikeValue(curSubset | bitmask(i,model.h));
    end

    % Which value of -log2(likelihood) is lowest?
    [minNegLogLike indexMin] = min(newNegLogLikes);
    assert(minNegLogLike<Inf,'curSubset has no 0s in it!');
    bestNonfeatureCostSizeK = minNegLogLike + ...
        model.codingCost + ...
        model.modelCostOfK(k);

    % Update our current subset and, possibly, our best score so far.
    curSubset = curSubset | bitmask(indexMin, model.h);
    if(bestNonfeatureCostSizeK < lowestNonfeatureCostEver)
        candidateNewBetaHat = newCandidateBetaHats{indexMin};
        lowestNonfeatureCostEver = bestNonfeatureCostSizeK;
        bestNegLogLike = minNegLogLike;
    else
        if(stopEarly) break; end
    end
end

% If no subset for this feature beat the best cost ever seen for any of the
% candidate features, then return junk.
if(candidateNewBetaHat==NONE_CONST)
    nonFeatureCostIfAdded = Inf; % It's not really, but we want to avoid adding this feature.
    bestNegLogLike = Inf;
else
    nonFeatureCostIfAdded = lowestNonfeatureCostEver;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function out = bitmask(i,h)
% bitmask
%  Produces a 1 x h vector that has a 1 at position i and zeros
%   everywhere else.
%  Input:
%   i: The position where the 1 should go.
%   h: Length of the vector.
assert(i <= h, 'i > h.');
out = zeros(1,h);
out(i) = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function print_diagnostics(model,bestFeatureToAdd,cur_k,prevLikelihoodCost,prevCodingCost,varargin)
% Print important info about the current state of the model in mic.m.

% Parse input.
parser = inputParser;
parser.FunctionName = 'print_diagnostics';
parser.addRequired('model');
parser.addRequired('bestFeatureToAdd');
parser.addRequired('cur_k');
parser.addRequired('prevLikelihoodCost');
parser.addRequired('prevCodingCost');
parser.parse(model,bestFeatureToAdd,cur_k,prevLikelihoodCost,prevCodingCost,varargin{:});
model = parser.Results.model;
bestFeatureToAdd = parser.Results.bestFeatureToAdd;
cur_k = parser.Results.cur_k;
prevLikelihoodCost = parser.Results.prevLikelihoodCost;
prevCodingCost = parser.Results.prevCodingCost;

% Print.
fprintf('#feat=%3i,  adding feat. %4i with %2i resp.,  like.=%4.0f,  cod.=%3.0f,  chg.like.=%5.0f,  chg.code=%5.0f,  avg.sig.=%5.4f\n',...
    model.nFeatures,...
    bestFeatureToAdd,...
    cur_k,...
    model.likelihoodCost,...
    model.codingCost,...
    model.likelihoodCost-prevLikelihoodCost,...
    model.codingCost-prevCodingCost,...
    mean(diag(model.curNullSigmaHat)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nearPD.m
%  This function is a Matlab rip-off of an R function called 'nearPD' in
%  the Matrix package:
%  http://rweb.stat.umn.edu/R/library/Matrix/html/nearPD.html
%  The complete text of the original R function appears below in a comment.
function X = nearPD(X, varargin)

% Parse input.
parser = inputParser;
parser.FunctionName = 'nearPD';
parser.addRequired('X', @isnumeric);
parser.addParamValue('maxit', 5000, @(x)x>=1);
parser.addParamValue('isCorrMatrix', 1, @(x)or(x==0,x==1));
parser.parse(X, varargin{:});
X = parser.Results.X;
maxit = parser.Results.maxit;
isCorrMatrix = parser.Results.isCorrMatrix;

% Get sizes
[n n2] = size(X);
assert(n==n2,'X is not square.');

% Check symmetry.
SMALL_NUM = .0001;
assert(all(all(X-X'<SMALL_NUM)),'X is not symmetric.');

% Set params.
DO2EIGEN = true;
EIG_TOL = 1e-6;
CONV_TOL = .001; %1e-7;
POSD_TOL = 1e-8;

% Create U.
U = zeros(n);

% Set iteration params.
iter = 0;
converged = false;
conv = Inf;

% Iterate.
while(iter < maxit && ~converged)
   Y = X;
   T = Y - U;
   
   % Project onto PSD matrices.
   [Q,D] = eig(X);
   d = diag(D);
   
   % Create mask.
   p = d > EIG_TOL * d(1);
   
   % Use p mask.
   % NOTE: If this code runs slowly, see if there's a way to speed it up
   % like in the R function....
   Q = Q(:,p);
   X = Q * D(p,p) * Q';
   
   % Update Dykstra's correction.
   U = X - T;
   
   % Project onto symmetric and possibly 'given diag' matrices.
   X = (X + X')/2;
   if(isCorrMatrix)
       X(eye(n)==1)=1;
   end
   
   conv = norm(Y-X,inf) / norm(Y, inf);
   iter = iter + 1;
   
   converged = (conv <= CONV_TOL);
end

if(~converged)
    warning('nearPD did not converge in %i iterations.',iter);
end

if(DO2EIGEN)
   [Q,D] = eig(X);
   
   % Matlab reverses the eigenvalues from the way R does things.
   Q = fliplr(Q);
   d = fliplr(diag(D)');
   
   Eps = POSD_TOL * abs(d(1)); % The R code says to take d(1), but Matlab reverses the order of eigenvalues.
   if(d(n) < Eps)
      d(d < Eps) = Eps;
      oDiag = diag(X)';
      X = Q * (repmat(d',1,n) .* Q');
      D = sqrt(max(Eps, oDiag) ./ diag(X)');
      X = repmat(D',1,n) .* X .* repmat(D,n,1);
   end
end

if(isCorrMatrix)
    X(eye(n)==1)=1;
end

assert(all(eig(X)>0),'The X returned by nearPD is not positive definite!');


%{
## nearcor.R :
## Copyright (2007) Jens Oehlschlägel
## GPL licence, no warranty, use at your own risk

nearPD <-
    ## Computes the nearest correlation matrix to an approximate
    ## correlation matrix, i.e. not positive semidefinite.

    function(x               # n-by-n approx covariance/correlation matrix
             , corr = FALSE, keepDiag = FALSE
             , do2eigen = TRUE  # if TRUE do a sfsmisc::posdefify() eigen step
             , only.values = FALSE# if TRUE simply return lambda[j].
             , eig.tol   = 1e-6 # defines relative positiveness of eigenvalues compared to largest
             , conv.tol  = 1e-7 # convergence tolerance for algorithm
             , posd.tol  = 1e-8 # tolerance for enforcing positive definiteness
             , maxit    = 100 # maximum number of iterations allowed
             , trace = FALSE # set to TRUE (or 1 ..) to trace iterations
             )
{
    stopifnot(isSymmetric(x))
    n <- ncol(x)
    if(keepDiag) diagX0 <- diag(x)
    ## U should be like x, but filled with '0' -- following also works for 'Matrix':
    U <- x; U[] <- 0
    X <- x
    iter <- 0 ; converged <- FALSE; conv <- Inf

    while (iter < maxit && !converged) {
        Y <- X
        T <- Y - U

        ## project onto PSD matrices
        e <- eigen(Y, symmetric = TRUE)
        Q <- e$vectors
        d <- e$values
        ## D <- diag(d)

        ## create mask from relative positive eigenvalues
        p <- d > eig.tol*d[1]

        ## use p mask to only compute 'positive' part
        Q <- Q[,p,drop = FALSE]
        ## X <- Q %*% D[p,p,drop = FALSE] %*% t(Q)  --- more efficiently :
        X <- tcrossprod(Q * rep(d[p], each=nrow(Q)), Q)

        ## update Dykstra's correction
        U <- X - T

        ## project onto symmetric and possibly 'given diag' matrices:
        X <- (X + t(X))/2
        if(corr) diag(X) <- 1
        else if(keepDiag) diag(X) <- diagX0

        conv <- norm(Y-X, "I") / norm(Y, "I")
        iter <- iter + 1
	if (trace)
	    cat(sprintf("iter %3d : #{p}=%d, ||Y-X|| / ||Y||= %11g\n",
			iter, sum(p), conv))

        converged <- (conv <= conv.tol)
    }

    if(!converged) {
        warning("nearPD() did not converge in ", iter, " iterations")
    }

    ## force symmetry is *NEVER* needed, we have symmetric X here!
    ## X <- (X + t(X))/2
    if(do2eigen || only.values) {
        ## begin from posdefify(sfsmisc)
        e <- eigen(X, symmetric = TRUE)
        d <- e$values
        Eps <- posd.tol * abs(d[1])
        if (d[n] < Eps) {
            d[d < Eps] <- Eps
            if(!only.values) {
                Q <- e$vectors
                o.diag <- diag(X)
                X <- Q %*% (d * t(Q))
                D <- sqrt(pmax(Eps, o.diag)/diag(X))
                X[] <- D * X * rep(D, each = n)
            }
        }
        if(only.values) return(d)

        ## unneeded(?!): X <- (X + t(X))/2
        if(corr) diag(X) <- 1
        else if(keepDiag) diag(X) <- diagX0

    } ## end from posdefify(sfsmisc)

    structure(list(mat =
		   new("dpoMatrix", x = as.vector(X),
		       Dim = c(n,n), Dimnames = .M.DN(x)),
                   eigenvalues = d,
                   corr = corr, normF = norm(x-X, "F"), iterations = iter,
		   rel.tol = conv, converged = converged),
	      class = "nearPD")
}
%}