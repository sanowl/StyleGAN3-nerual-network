class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) class for model parameters.
    """
    def __init__(self, parameters, decay):
        self.parameters = list(parameters)
        self.decay = decay
        self.shadow_params = [p.clone().detach() for p in self.parameters]

    def update(self, parameters):
        for shadow_param, param in zip(self.shadow_params, parameters):
            shadow_param.data = self.decay * shadow_param.data + (1.0 - self.decay) * param.data