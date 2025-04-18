from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import bambi as bmb
import bambi.priors as priors
from scipy.special import expit
import arviz as az
from statsmodels.formula.api import logit as bayes_logit
import matplotlib.pyplot as plt


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

    def update(self, input_data):
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
    model.trigger = AccuracyThreshold(model=model, threshold=0.7)       
    """
    
    def __init__(self, predictColName='prediction', outcomeColName='outcome', dateCol='date'):
        super(RecalibratePredictions, self).__init__()
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
    
    def update(self, input_data):
        # Get predictions
        preds = self.predict(input_data)
        
        # Convert to linear predictor scale
        lp = self.__inverseSigmoid(preds)
        
        # Work out model calibration
        logreg = LogisticRegression(penalty=None, max_iter=1000) # 'l1', 'elasticnet', 'l2' or None
        logreg.fit(np.array(lp).reshape(-1, 1), input_data[self.outcomeColName].astype(int))
        intercept = logreg.intercept_
        scale = logreg.coef_[0]
        
        # Add hook to adjust predictions accordingly
        recal = lambda p: self.__sigmoid(self.__inverseSigmoid(p) * scale + intercept)
        self.addPostPredictHook(recal)


    def CalculateControlLimits(self, input_data, startCLDate, endCLDate, warningCL, recalCL, warningSDs, recalSDs):
        """Calculate the static control limits of data using either the specific period (startCLDate to endCLDate), 
        the inputted control limits (warningCL, recalCL), or the first X months (nummMonths) since the start of the 
        data.

        Args:
            input_data (pd.DataFrame): The input data for updating the model.
            startCLDate (str): Start date to determine control limits from. Defaults to None.
            endCLDate (str): End date to determine control limits from. Defaults to None.
            warningCL (float): A manually set control limit for the warning control limit.
            recalCL (float): A manually set control limit for the recalibration trigger limit.
            warningSDs (int or float): Number of standard deviations from the mean to set the warning limit to. Defaults to 2.
            recalSDs (int or float): Number of standard deviations from the mean to set the recalibration trigger to. Defaults to 3.

        Returns:
            float, float: Two upper control limits for the warning and danger/recalibration trigger zones.
        """
        def CalculateError(group):
            predictions = group['predictions']
            differences = group[self.outcomeColName] - predictions
            sum_of_differences = np.sum(differences)/len(group[self.outcomeColName])
            return sum_of_differences
        
        if startCLDate is not None and endCLDate is not None:
            # Get the logreg error at each timestep within the control limit determination period
            createCLdf = input_data[(input_data[self.dateCol] >= startCLDate) & (input_data[self.dateCol] <= endCLDate)].copy()
            
            # Predictions column
            createCLdf['predictions'] = self.predict(createCLdf)

            errors_by_date = createCLdf.groupby(self.dateCol).apply(CalculateError)
            self.mean_error = errors_by_date.mean()
            std_dev_error = errors_by_date.std()
            
            self.u2sdl = self.mean_error + warningSDs * std_dev_error
            self.u3sdl = self.mean_error + recalSDs * std_dev_error
            self.l2sdl = self.mean_error - warningSDs * std_dev_error
            self.l3sdl = self.mean_error - recalSDs * std_dev_error

        #elif warningCL is not None and recalCL is not None:
        else:
            # Predictions column
            input_data['predictions'] = self.predict(input_data)
            errors_by_date = input_data.groupby(self.dateCol).apply(CalculateError)
            self.mean_error = errors_by_date.mean()
            std_dev_error = errors_by_date.std()
            self.u3sdl = recalCL
            self.u2sdl = warningCL
            self.l3sdl = -recalCL
            self.l2sdl = -warningCL
    
        return self.u2sdl, self.u3sdl, self.l2sdl, self.l3sdl

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
                tune=1000, cores=1, chains=4, model_formula=None):
        super(BayesianModel, self).__init__()
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

        generate_priors = False
        for _, value in self.priors.items():
            if value is None:
                generate_priors = True


        if generate_priors:
            if input_data is None:
                raise ValueError("No input data provided to generate priors from.")
            faux_model = bayes_logit(self.model_formula, data=input_data).fit()

            self.priors = {}
            for idx in range(0, len(self.coef_names)):
                self.priors[faux_model.params.index[idx]] = (faux_model.params[idx], faux_model.bse[idx])

        
    def predict(self, input_data):
        if self.bayes_model is None:
            preds = input_data[self.predictColName]
        else:
            idata = self.bayes_model.predict(data=input_data, idata=self.inference_data, inplace=False)
            mean_results = az.summary(idata.posterior)
            number_coefs = len(self.coef_names)
            df_filtered = mean_results.iloc[number_coefs:]
            preds = df_filtered["mean"].values.flatten()
            # preds = az.summary(preds.posterior["outcome_mean"])["mean"].values.flatten()
            preds = pd.Series(preds, index=input_data[self.predictColName].index)
            preds = preds.clip(1e-10, 1 - 1e-10) # clip to prevent log(0) or log(1) errors
        return preds


    
    def update(self, input_data):
        if self.verbose:
            print("\n*** PRIORS ***")
        bmb_priors = {}
        for prior_key, prior_values in self.priors.items():
            prior_mean, prior_std = prior_values
            bmb_priors[prior_key] = bmb.Prior("Normal", mu=prior_mean, sigma=prior_std)

            if self.verbose:
                print(f"{prior_key} mean coef: {prior_mean:.2f} ± {prior_std:.2f}")

        self.bayes_model = bmb.Model(self.model_formula, data=input_data, family="bernoulli", priors=bmb_priors)
            

        self.inference_data = self.bayes_model.fit(draws=self.draws, tune=self.tune, cores=self.cores, chains=self.chains)
        posterior_samples = self.inference_data.posterior 

        if self.verbose:
            print("\n*** POSTERIORS ***")
        if self.plot_idata:
            az.plot_trace(self.inference_data, figsize=(10, 7), )
        

        # Update the priors for the next run of the Bayesian model
        self.priors = {
            predictor: (
                posterior_samples[predictor].values.flatten().mean(),
                posterior_samples[predictor].values.flatten().std()
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