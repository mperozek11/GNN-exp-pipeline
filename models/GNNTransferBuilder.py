from statistics import mode


class GNNTransferBuilder:


    def __init__(model_class, model_state_dict, self):
        self.model_class = model_class
        self.model_state_dict = model_state_dict
        # freeze layers 
    
    # adds a linear classifier as the last layer of the model to be trained on desired data
    def get_transfer_model(self, out_dim):
        pass