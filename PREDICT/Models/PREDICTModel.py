class PREDICTModel:
    """
    A class used to represent the PREDICT Model.

    Methods
    -------
    __init__()
        Initializes the model with pre and post prediction hooks.
    predict(input_data)
        Makes predictions based on the input data.
    trigger(input_data)
        Evaluates whether the model needs to be updated.
    update(input_data)
        Updates the model if required.
    addPrePredictHook(hook)
        Adds a hook to be executed before making predictions.
    addPostPredictHook(hook)
        Adds a hook to be executed after making predictions.
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

        Parameters
        ----------
        input_data : any
            The input data for making predictions.
        """
        pass

    def trigger(self, input_data):
        """
        Evaluates whether the model needs to be updated based on the input data.

        Parameters
        ----------
        input_data : any
            The input data to evaluate the model update.

        Returns
        -------
        bool
            Returns False indicating no update is required.
        """
        return False

    def update(self, input_data):
        """
        Updates the model if required based on the input data.

        Parameters
        ----------
        input_data : any
            The input data for updating the model.
        """
        pass

    def addPrePredictHook(self, hook):
        """
        Adds a hook to be executed before making predictions.

        Parameters
        ----------
        hook : callable
            A function to be executed before predictions.
        """
        self.prePredictHooks.append(hook)
        
    def addPostPredictHook(self, hook):
        """
        Adds a hook to be executed after making predictions.

        Parameters
        ----------
        hook : callable
            A function to be executed after predictions.
        """
        self.postPredictHooks.append(hook)
