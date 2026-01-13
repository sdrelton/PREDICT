from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import bambi as bmb
import bambi.priors as priors
from scipy.special import expit
import arviz as az
from statsmodels.formula.api import logit as bayes_logit
import matplotlib.pyplot as plt
import logging


class PREDICTModel:
    """
    A class used to represent the PREDICT Model.
    """

    def __init__(self):
        """
        Initializes the model with empty lists for pre and post prediction hooks.
        """
        self.prePredictHooks = list()
        self.postPredictHooks = list()

    def predict(self, input_data):
        """
        Makes predictions based on the input data.

        Args:
            input_data (any): The input data for making predictions.
        """
        pass

    def trigger(self, input_data):
        """
        Evaluates whether the model needs to be updated based on the input data.

        Args: 
            input_data (any): The input data to evaluate the model update.

        Returns:
            bool: Returns False indicating no update is required.
        """
        return False

    def update(self, input_data, windowStart, windowEnd, recalPeriod):
        """
        Updates the model if required based on the input data.

        Args:
            input_data (any): The input data for updating the model.
        """
        pass

    def addPrePredictHook(self, hook):
        """
        Adds a hook to be executed before making predictions.

        Args:
            hook (callable): A function to be executed before predictions.
        """
        self.prePredictHooks.append(hook)
        
    def addPostPredictHook(self, hook):
        """
        Adds a hook to be executed after making predictions.

        Args:
            hook (callable): A function to be executed after predictions.
        """
        self.postPredictHooks.append(hook)
        


class EvaluatePredictions(PREDICTModel):
    """
    A class used to evaluate the predictions arising from another model which are already in the dataframe.
    
    Args:
        colName (str): The name of the column in the dataframe containing the predictions (default='prediction').
    """
    def __init__(self, colName='prediction'):
        self.colName = colName
        
    def predict(self, input_data):
        return input_data[self.colName]
    
    
class RecalibratePredictions(PREDICTModel):
    """
    A class which recalibrates the predictions arising from another model based on the trigger function.
    
    Recalibration involves using a logistic regression to adjust the model predictions.
    
    Needs to be followed by setting a trigger function (see example).
    
    Args:
        predictColName (str): The name of the column in the dataframe containing the predictions (default='prediction').
        outcomeColName (str): The name of the column in the dataframe containing the outcomes (default='outcome').
        dateCol (str): The name of the column in the dataframe containing the dates (default='date').
        
    Examples
    --------
    # Create a model which recalibrates predictions when triggered
    # Full example can be found in Examples/recalibration_example.ipynb
    model = RecalibratePredictions()
    model.trigger = AccuracyThreshold(model=model, update_threshold=0.7)       
    """
    
    def __init__(self, predictColName='prediction', outcomeColName='outcome', dateCol='date'):
        super().__init__()
        self.predictColName = predictColName
        self.outcomeColName = outcomeColName
        self.dateCol = dateCol
        
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __inverseSigmoid(self, y):
        return np.log(y / (1 - y))
        
    def predict(self, input_data):
        preds = input_data[self.predictColName]
        # Recalibrate from any hooks that have been added
        for hook in self.postPredictHooks:
            preds = hook(preds)
        return preds
    
    def update(self, input_data, windowStart, windowEnd, recalPeriod):
        # Get predictions
        if recalPeriod is None:
            curdata = input_data[(input_data[self.dateCol] >= windowStart) & (input_data[self.dateCol] < windowEnd)]
        else:
            curdata = input_data[(input_data[self.dateCol] >= (windowEnd - pd.Timedelta(days=recalPeriod))) & (input_data[self.dateCol] < windowEnd)]
        
        preds = self.predict(curdata)
        
        # Convert to linear predictor scale
        lp = self.__inverseSigmoid(preds)
        
        # Work out model calibration
        logreg = LogisticRegression(penalty=None, max_iter=5000) # 'l1', 'elasticnet', 'l2' or None
        logreg.fit(np.array(lp).reshape(-1, 1), curdata[self.outcomeColName].astype(int))
        intercept = logreg.intercept_
        scale = logreg.coef_[0]
        
        # Add hook to adjust predictions accordingly
        recal = lambda p: self.__sigmoid(self.__inverseSigmoid(p) * scale + intercept)
        self.addPostPredictHook(recal)
        return intercept, scale

class BayesianModel(PREDICTModel):
    """ A class which uses Bayesian regression to update the coefficients of a logistic regression model.

    Args:
        priors (dict): Dictionary of predictors (key) and their mean and stds of the coefficients (values) e.g. {"blood_pressure": (2.193, 0.12)}, 
            this must include "Intercept" as a dictionary key.
            If any of the prior keys are None then the prior coefficients are estimated using a logistic regression model.
        input_data (pd.DataFrame): The input data for creating the model to calculate initial priors if none are provided.
        predictColName (str): The name of the column in the dataframe containing the predictions (default='prediction').
        outcomeColName (str): The name of the column in the dataframe containing the outcomes (default='outcome').
        dateCol (str): The name of the column in the dataframe containing the dates (default='date').
        verbose (bool): Whether to print the priors and posteriors of the model (default=True).
        plot_idata (bool): Whether to plot the inference_data trace plot (default=False).
        draws (int): Number of draws to use in the Bayesian model (default=1000).
        tune (int): Number of tuning steps to use in the Bayesian model (default=1000).
        cores (int): Number of cores to use in the Bayesian model (default=1).
        chains (int): Number of chains to use in the Bayesian model (default=4).
        model_formula (str): The formula to use in the Bayesian model (default=None, if None then a standard linear model formula is used without interactions).

    Raises:
        ValueError: If the priors are not in a dictionary format.
        ValueError: If the required 'Intercept' is missing from the priors.keys().
        ValueError: If priors are not provided and input_data is None.

    Example
    --------
    # Create a Bayesian model which refits and gives new coefficients and predictions when triggered.
    # Full example can be found in Examples/bayesian_example.ipynb
    model = BayesianModel(input_data=df, priors={"Intercept": (0.34, 0.1), "age": (1.56, 0.4), "systolic_bp": (5.34, 0.2)})
    model.trigger = BayesianRefitTrigger(model=model, input_data=df, refitFrequency=1)
    """
    def __init__(self, priors, input_data=None, predictColName='prediction', outcomeColName='outcome', dateCol='date', verbose=True, plot_idata=False, draws=1000,
                tune=250, cores=1, chains=4, model_formula=None):
        super().__init__()
        self.predictColName = predictColName
        self.outcomeColName = outcomeColName
        self.dateCol = dateCol
        self.priors = priors
        self.verbose = verbose
        self.plot_idata = plot_idata
        self.draws = draws
        self.tune = tune
        self.cores = cores
        self.chains = chains
        self.model_formula = model_formula
        self.bayes_model = None
        self.inference_data = None

        if not isinstance(self.priors, dict):
            raise ValueError("Provided priors are not in a dictionary format. Either provide no priors or provide them in a dictionary form e.g. {'blood_pressure': (2.193, 0.12)} ")
        
        
        self.coef_names = list(self.priors.keys())
        self.predictors = self.coef_names.copy()
        self.predictors.remove("Intercept")
        if "Intercept" not in self.coef_names:
            raise ValueError("The required 'Intercept' is missing from the priors.keys().")
        if self.model_formula is None:
            self.model_formula = self.outcomeColName + "~" + "+".join(self.predictors)
            print("No model formula was provided, using standard linear model formula.")
        print("Model formula is set to: ", self.model_formula)
        generate_priors = False
        for _, value in self.priors.items():
            if value is None:
                generate_priors = True


        if generate_priors:
            if input_data is None:
                raise ValueError("No input data provided to generate priors from.")
            print("Generating priors from input data...")
            faux_model = bayes_logit(self.model_formula, data=input_data).fit()

            self.priors = {}
            for idx in range(0, len(self.coef_names)):
                self.priors[faux_model.params.index[idx]] = (faux_model.params[idx], faux_model.bse[idx])

        
    def predict(self, input_data):
        if self.bayes_model is None:
            preds = input_data[self.predictColName]
            if self.verbose:
                logging.info("No Bayesian model has been fitted yet. Returning the predictions from the input data.")
        else:
            idata = self.bayes_model.predict(data=input_data, idata=self.inference_data, inplace=False)

            # Get mean posterior predictions across chains and draws
            pred_array = idata.posterior["p"].mean(dim=["chain", "draw"]).values.flatten()

            # Sanity check to avoid future mismatches
            assert len(pred_array) == len(input_data), f"Mismatched lengths: {len(pred_array)} vs {len(input_data)}"

            # Final formatted predictions
            preds = pd.Series(np.clip(pred_array, 1e-10, 1 - 1e-10), index=input_data.index)
            
        return preds


    
    def update(self, input_data, windowStart, windowEnd, recalPeriod):
        if self.verbose:
            print("\n*** PRIORS ***")
        bmb_priors = {}
        for prior_key, prior_values in self.priors.items():
            prior_mean, prior_std = prior_values
            bmb_priors[prior_key] = bmb.Prior("Normal", mu=prior_mean, sigma=prior_std)

            if self.verbose:
                print(f"{prior_key} mean coef: {prior_mean:.2f} ± {prior_std:.2f}")

        curdata = input_data[(input_data[self.dateCol] >= windowStart) & (input_data[self.dateCol] < windowEnd)]
        self.bayes_model = bmb.Model(self.model_formula, data=curdata, family="bernoulli", priors=bmb_priors)
            

        self.inference_data = self.bayes_model.fit(draws=self.draws, tune=self.tune, cores=self.cores, chains=self.chains, max_treedepth=10, target_accept=0.9)#, inference_method='mcmc')
        posterior_samples = self.inference_data.posterior 

        if self.verbose:
            print("\n*** POSTERIORS ***")
        if self.plot_idata:
            az.plot_trace(self.inference_data, figsize=(10, 7), )
        

        # Update only the prior means, keep original stds to prevent narrowing intervals
        self.priors = {
            predictor: (
                posterior_samples[predictor].values.flatten().mean(),
                self.priors[predictor][1]  # retain original std
            )
            for predictor in self.coef_names
        }

        self.posterior_samples = {
            predictor: (
                posterior_samples[predictor].values.flatten()
            )
            for predictor in self.coef_names
        }


        if self.verbose:
            print(f'Intercept mean coef: {posterior_samples["Intercept"].values.flatten().mean():.2f} ± {posterior_samples["Intercept"].values.flatten().std():.2f}')
            for predictor in self.predictors:
                coef_mean = posterior_samples[predictor].values.flatten().mean()
                coef_std = posterior_samples[predictor].values.flatten().std()
                print(f"{predictor} mean coef: {coef_mean:.2f} ± {coef_std:.2f}")
        
    def get_coefs(self):
        """Return the current coefficients of the model for the loghook.

        Returns:
            dict: Current priors of the model.
        """
        return self.priors