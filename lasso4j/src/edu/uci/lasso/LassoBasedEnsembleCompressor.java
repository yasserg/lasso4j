package edu.uci.lasso;

import java.util.Random;

import edu.uci.jforests.dataset.Dataset;
import edu.uci.jforests.ensembles.Ensemble;
import edu.uci.jforests.ensembles.RegressionTree;
import edu.uci.jforests.util.MathUtil;

/*
 * This implemenation is based on:
 * Friedman, J., Hastie, T. and Tibshirani, R. (2008) Regularization
 * Paths for Generalized Linear Models via Coordinate Descent. 
 * http://www-stat.stanford.edu/~hastie/Papers/glmnet.pdf
 * 
 * @author: Yasser Ganjisaffar
 */

public class LassoBasedEnsembleCompressor
{
    // This module shouldn't consume more than 4GB of memory
    private static final long MAX_AVAILABLE_MEMORY = 4L * 1024 * 1024 * 1024;

    // In order to speed up the compression, we limit the number of observations,
    // but this limit is dependent on the number of features that we should learn 
    // their weights. In other words, for learning weights of more features, we 
    // need more observations.
    private static final int MAX_OBSERVATIONS_TO_FEATURES_RATIO = 100;

    // Number of relevance levels: Perfect, Excellent, Good, Fair, Bad
    private static final int NUM_RELEVANCE_LEVELS = 5;

    private static final double EPSILON = 1.0e-6;

    // The default number of lambda values to use
    private static final int DEFAULT_NUMBER_OF_LAMBDAS = 100;

    // Convergence threshold for coordinate descent
    // Each inner coordination loop continues until the relative change
    // in any coefficient is less than this threshold
    private static final double CONVERGENCE_THRESHOLD = 1.0e-4;

    private static final double SMALL = 1.0e-5;
    private static final int MIN_NUMBER_OF_LAMBDAS = 5;
    private static final double MAX_RSQUARED = 0.999;

    private float[] _targets;
    private float[][] _observations;
    private int _numFeatures;
    private int _numObservations;

    private Dataset _trainSet;
    private Ensemble _compressedEnsemble;
    private int[] _sampleObservationIndices;
    private Random _rnd;

    public void Initialize(int numTrees, Dataset trainSet, int randomSeed)
    {
        _numFeatures = numTrees;
        _trainSet = trainSet;
        _rnd = new Random(randomSeed);

        int maxObservations = (int) (MAX_AVAILABLE_MEMORY / _numFeatures / (Float.SIZE / 8));
        _numObservations = Math.Min(_trainSet.NumDocs, maxObservations);
        if (_numObservations > MAX_OBSERVATIONS_TO_FEATURES_RATIO * _numFeatures)
        {
            _numObservations = MAX_OBSERVATIONS_TO_FEATURES_RATIO * _numFeatures;
        }
        DoLabelBasedSampling(trainSet);

        _observations = new float[_numObservations][];
        for (int t = 0; t < _numFeatures; t++)
        {
            _observations[t] = new float[_numObservations];
        }
    }

    private void DoLabelBasedSampling(Dataset trainSet)
    {
        if (_numObservations == _trainSet.NumDocs)
        {
            // No sampling                
            _sampleObservationIndices = null;                
        }
        else
        {
            _sampleObservationIndices = new int[_numObservations];
            int[] perLabelDocCount = new int[NUM_RELEVANCE_LEVELS];
            for (int d = 0; d < trainSet.NumDocs; d++)
            {
                perLabelDocCount[trainSet.Labels[d]]++;
            }
            List<KeyValuePair<short, int>> labelFreqList = new List<KeyValuePair<short, int>>();
            for (short i = 0; i < NUM_RELEVANCE_LEVELS; i++)
            {
                labelFreqList.Add(new KeyValuePair<short,int>(i, perLabelDocCount[i]));
            }
            labelFreqList.Sort(delegate(KeyValuePair<short, int> c1, KeyValuePair<short, int> c2)
            {
                return Comparer<double>.Default.Compare(c1.Value, c2.Value);
            });
            int remainedDocs = _numObservations;
            double[] perLabelSampleRate = new double[NUM_RELEVANCE_LEVELS];
            for (short i = 0; i < NUM_RELEVANCE_LEVELS; i++)
            {
                short curLabel = labelFreqList[i].Key;
                int currentMax = remainedDocs / (NUM_RELEVANCE_LEVELS - i);
                int selectedDocs = Math.Min(perLabelDocCount[curLabel], currentMax);
                perLabelSampleRate[curLabel] = (double) selectedDocs / perLabelDocCount[curLabel];
                remainedDocs -= selectedDocs;
            }
            int obsCount = 0;
            for (int d = 0; d < trainSet.NumDocs; d++)
            {
                if (_rnd.NextDouble() <= perLabelSampleRate[trainSet.Labels[d]])
                {
                    _sampleObservationIndices[obsCount] = d;
                    obsCount++;
                    if (obsCount == _numObservations)
                    {
                        break;
                    }
                }
            }
            // Since it's a random process, the generated number of observations might be
            // slightly different. So, we make them the same.
            _numObservations = obsCount;
        }
    }

    public void SetTreeScores(int idx, float[] scores)
    {
        if (_sampleObservationIndices == null)
        {
            for (int i = 0; i < scores.length; i++)
            {
            	_observations[idx][i] = scores[i];
            }
        }
        else
        {
            for (int i = 0; i < _numObservations; i++)
            {
            	_observations[idx][i] = scores[_sampleObservationIndices[i]];
            }
        }
    }

    private LassoFit GetLassoFit(int maxAllowedFeaturesPerModel)
    {
        long startTime = System.currentTimeMillis();

        if (maxAllowedFeaturesPerModel < 0)
        {
            maxAllowedFeaturesPerModel = _numFeatures;
        }
        int numberOfLambdas = DEFAULT_NUMBER_OF_LAMBDAS;
        int maxAllowedFeaturesAlongPath = (int)Math.min(maxAllowedFeaturesPerModel * 1.2, _numFeatures);

        // lambdaMin = flmin * lambdaMax
        double flmin = (_numObservations < _numFeatures ? 5e-2 : 1e-4);

        /********************************
        * Standardize predictors and target:
        * Center the target and features (mean 0) and normalize their vectors to have the same 
        * standard deviation
        */
        double[] featureMeans = new double[_numFeatures];
        double[] featureStds = new double[_numFeatures];
        double[] feature2residualCorrelations = new double[_numFeatures];

        float factor = (float)(1.0 / Math.sqrt(_numObservations));
        for (int j = 0; j < _numFeatures; j++)
        {
            double mean = MathUtil.getAvg(_observations[j]);
            featureMeans[j] = mean;
            for (int i = 0; i < _numObservations; i++)
            {
            	_observations[j][i] = (float)(factor * (_observations[j][i] - mean));
            }
            featureStds[j] = Math.sqrt(MathUtil.getDotProduct(_observations[j], _observations[j]));

            MathUtil.divideInPlace(_observations[j], (float)featureStds[j]);
        }

        float targetMean = (float)MathUtil.getAvg(_targets);
        for (int i = 0; i < _numObservations; i++)
        {
        	_targets[i] = factor * (_targets[i] - targetMean);
        }
        float targetStd = (float)Math.sqrt(MathUtil.getDotProduct(_targets, _targets));
        MathUtil.divideInPlace(_targets, targetStd);

        for (int j = 0; j < _numFeatures; j++)
        {
            feature2residualCorrelations[j] = MathUtil.getDotProduct(_targets, _observations[j]);
        }

        double[][] feature2featureCorrelations = MathUtil.allocateDoubleMatrix(_numFeatures, maxAllowedFeaturesAlongPath);
        double[] activeWeights = new double[_numFeatures];
        int[] correlationCacheIndices = new int[_numFeatures];
        double[] denseActiveSet = new double[_numFeatures];

        LassoFit fit = new LassoFit(numberOfLambdas, maxAllowedFeaturesAlongPath, _numFeatures);
        fit.numberOfLambdas = 0;

        double alf = Math.pow(Math.max(EPSILON, flmin), 1.0 / (numberOfLambdas - 1));
        double rsquared = 0.0;
        fit.numberOfPasses = 0;
        int numberOfInputs = 0;
        int minimumNumberOfLambdas = Math.min(MIN_NUMBER_OF_LAMBDAS, numberOfLambdas);

        double curLambda = 0;
        double maxDelta;
        for (int iteration = 1; iteration <= numberOfLambdas; iteration++)
        {
            System.out.println("Starting iteration " + iteration + " of Compression.");

            /**********
            * Compute lambda for this round
            */
            if (iteration == 1)
            {
                curLambda = Double.MAX_VALUE; // first lambda is infinity
            }
            else if (iteration == 2)
            {
                curLambda = 0.0;
                for (int j = 0; j < _numFeatures; j++)
                {
                    curLambda = Math.max(curLambda, Math.abs(feature2residualCorrelations[j]));
                }
                curLambda = alf * curLambda;
            }
            else
            {
                curLambda = curLambda * alf;
            }

            double prevRsq = rsquared;
            double v;
            while (true)
            {
                fit.numberOfPasses++;
                maxDelta = 0.0;
                for (int k = 0; k < _numFeatures; k++)
                {
                    double prevWeight = activeWeights[k];
                    double u = feature2residualCorrelations[k] + prevWeight;
                    v = (u >= 0 ? u : -u) - curLambda;
                    // Computes sign(u)(|u| - curLambda)+
                    activeWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);

                    // Is the weight of this variable changed?
                    // If not, we go to the next one
                    if (activeWeights[k] == prevWeight)
                    {
                        continue;
                    }

                    // If we have not computed the correlations of this
                    // variable with other variables, we do this now and
                    // cache the result
                    if (correlationCacheIndices[k] == 0)
                    {
                        numberOfInputs++;
                        if (numberOfInputs > maxAllowedFeaturesAlongPath)
                        {
                            // we have reached the maximum 
                            break;
                        }
                        for (int j = 0; j < _numFeatures; j++)
                        {
                            // if we have already computed correlations for
                            // the jth variable, we will reuse it here.
                            if (correlationCacheIndices[j] != 0)
                            {
                                feature2featureCorrelations[j][numberOfInputs - 1] = feature2featureCorrelations[k][correlationCacheIndices[j] - 1];
                            }
                            else
                            {
                                // Correlation of variable with itself if one
                                if (j == k)
                                {
                                    feature2featureCorrelations[j][numberOfInputs - 1] = 1.0;
                                }
                                else
                                {
                                    feature2featureCorrelations[j][numberOfInputs - 1] = MathUtil.getDotProduct(_observations[j], _observations[k]);
                                }
                            }
                        }
                        correlationCacheIndices[k] = numberOfInputs;
                        fit.indices[numberOfInputs - 1] = k;
                    }

                    // How much is the weight changed?
                    double delta = activeWeights[k] - prevWeight;
                    rsquared += delta * (2.0 * feature2residualCorrelations[k] - delta);
                    maxDelta = Math.max((delta >= 0 ? delta : -delta), maxDelta);

                    for (int j = 0; j < _numFeatures; j++)
                    {
                        feature2residualCorrelations[j] -= feature2featureCorrelations[j][correlationCacheIndices[k] - 1] * delta;
                    }
                }

                if (maxDelta < CONVERGENCE_THRESHOLD || numberOfInputs > maxAllowedFeaturesAlongPath)
                {
                    break;
                }

                for (int ii = 0; ii < numberOfInputs; ii++)
                {
                    denseActiveSet[ii] = activeWeights[fit.indices[ii]];
                }

                do
                {
                    fit.numberOfPasses++;
                    maxDelta = 0.0;
                    for (int l = 0; l < numberOfInputs; l++)
                    {
                        int k = fit.indices[l];
                        double prevWeight = activeWeights[k];
                        double u = feature2residualCorrelations[k] + prevWeight;
                        v = (u >= 0 ? u : -u) - curLambda;
                        activeWeights[k] = (v > 0 ? (u >= 0 ? v : -v) : 0.0);
                        if (activeWeights[k] == prevWeight)
                        {
                            continue;
                        }
                        double delta = activeWeights[k] - prevWeight;
                        rsquared += delta * (2.0 * feature2residualCorrelations[k] - delta);
                        maxDelta = Math.max((delta >= 0 ? delta : -delta), maxDelta);
                        for (int j = 0; j < numberOfInputs; j++)
                        {
                            feature2residualCorrelations[fit.indices[j]] -= feature2featureCorrelations[fit.indices[j]][correlationCacheIndices[k] - 1] * delta;
                        }
                    }
                } while (maxDelta >= CONVERGENCE_THRESHOLD);

                for (int ii = 0; ii < numberOfInputs; ii++)
                {
                    denseActiveSet[ii] = activeWeights[fit.indices[ii]] - denseActiveSet[ii];
                }
                for (int j = 0; j < _numFeatures; j++)
                {
                    if (correlationCacheIndices[j] == 0)
                    {
                        feature2residualCorrelations[j] -= MathUtil.getDotProduct(denseActiveSet, feature2featureCorrelations[j], numberOfInputs);
                    }
                }
            }

            if (numberOfInputs > maxAllowedFeaturesAlongPath)
            {
                break;
            }
            if (numberOfInputs > 0)
            {
                for (int ii = 0; ii < numberOfInputs; ii++)
                {
                    fit.compressedWeights[iteration - 1][ii] = activeWeights[fit.indices[ii]];
                }
            }
            fit.numberOfWeights[iteration - 1] = numberOfInputs;
            fit.rsquared[iteration - 1] = rsquared;
            fit.lambdas[iteration - 1] = curLambda;
            fit.numberOfLambdas = iteration;

            if (iteration < minimumNumberOfLambdas)
            {
                continue;
            }

            int me = 0;
            for (int j = 0; j < numberOfInputs; j++)
            {
                if (fit.compressedWeights[iteration - 1][j] != 0.0)
                {
                    me++;
                }
            }
            if (me > maxAllowedFeaturesPerModel || ((rsquared - prevRsq) < (SMALL * rsquared)) || rsquared > MAX_RSQUARED)
            {
                break;
            }
        }

        for (int k = 0; k < fit.numberOfLambdas; k++)
        {
            fit.lambdas[k] = targetStd * fit.lambdas[k];
            int nk = fit.numberOfWeights[k];
            for (int l = 0; l < nk; l++)
            {
                fit.compressedWeights[k][l] = targetStd * fit.compressedWeights[k][l] / featureStds[fit.indices[l]];
                if (fit.compressedWeights[k][l] != 0)
                {
                    fit.nonZeroWeights[k]++;
                }
            }
            double product = 0;
            for (int i = 0; i < nk; i++)
            {
                product += fit.compressedWeights[k][i] * featureMeans[fit.indices[i]];
            }
            fit.intercepts[k] = targetMean - product;
        }

        // First lambda was infinity; fixing it
        fit.lambdas[0] = Math.exp(2 * Math.log(fit.lambdas[1]) - Math.log(fit.lambdas[2]));

        long duration = System.currentTimeMillis() - startTime;
        System.out.println("Elapsed time for compression: " + duration);

        return fit;
    }

    private Ensemble GetEnsembleFromSolution(LassoFit fit, int solutionIdx, Ensemble originalEnsemble)
    {
        Ensemble ensemble = new Ensemble();
        int weightsCount = fit.numberOfWeights[solutionIdx];
        for (int i = 0; i < weightsCount; i++)
        {
            double weight = fit.compressedWeights[solutionIdx][i];
            if (weight != 0)
            {
                RegressionTree tree = originalEnsemble.getTreeAt(fit.indices[i]);
                ensemble.addTree(tree, weight);
            }
        }
        return ensemble;
    }

    private void LoadTargets(double[] trainScores, int bestIteration)
    {
        if (bestIteration == -1)
        {
            bestIteration = _numFeatures;
        }
        double[] targetScores;
        if (bestIteration == _numFeatures)
        {
            // If best iteration is the last one, train scores will be our targets
            targetScores = trainScores;
        }
        else
        {
            // We need to sum up scores of trees before best iteration to find targets
            targetScores = new double[_numObservations];
            for (int d = 0; d < _numObservations; d++)
            {
                for (int t = 0; t < bestIteration; t++)
                {
                    targetScores[d] += _observations[t][d];
                }
            }
        }
        _targets = new float[_numObservations];
        if (_sampleObservationIndices == null || bestIteration != _numFeatures)
        {
            for (int i = 0; i < _numObservations; i++)
            {
            	_targets[i] = (float)targetScores[i];
            }
        }
        else
        {
            for (int i = 0; i < _numObservations; i++)
            {
            	_targets[i] = (float)targetScores[_sampleObservationIndices[i]];
            }
        }
    }

    public boolean Compress(Ensemble ensemble, double[] trainScores, int bestIteration, int maxTreesAfterCompression)
    {
        LoadTargets(trainScores, bestIteration);

        LassoFit fit = GetLassoFit(maxTreesAfterCompression);
        int numberOfSolutions = fit.numberOfLambdas;
        int bestSolutionIdx = 0;

        System.out.println("Compression R2 values:");
        for (int i = 0; i < numberOfSolutions; i++)
        {
        	System.out.println((i + 1) + "\t" + fit.nonZeroWeights[i] + "\t" + fit.rsquared[i]);                
        }
        bestSolutionIdx = numberOfSolutions - 1;
        _compressedEnsemble = GetEnsembleFromSolution(fit, bestSolutionIdx, ensemble);
        return true;
    }

    public Ensemble GetCompressedEnsemble()
    {
        return _compressedEnsemble;
    }
}
