import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os




# Classifier that takes a (batch_size, num_explanation, feature_dim) to make a prediction
class ExplanationFeatureConcatenatorClassifier(nn.Module):
    def __init__(self, num_classes, num_explanations, feature_dim, projection_dim, hidden_dim, num_layers, dropout, regex_features):
        super(ExplanationFeatureConcatenatorClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_explanations = num_explanations

        # first project down to a smaller latent space
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.regex_features = regex_features

        if projection_dim == 0:
            self.projection_dim = self.feature_dim
            self.projection_layer = lambda x: x
        else:
            self.projection_dim = projection_dim
            self.projection_layer = nn.Linear(self.feature_dim, self.projection_dim)
        if num_layers==0:
            modules = [nn.Linear(self.projection_dim*self.num_explanations + self.regex_features, self.num_classes)]
        else:
            modules = [nn.Linear(self.projection_dim*self.num_explanations + self.regex_features, self.hidden_dim), nn.ReLU()]
            if self.dropout > 0.0:
                modules.append(nn.Dropout(p=dropout))

            for layer in range(num_layers):
                modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                 #TODO add dropout layer here
                if self.dropout > 0.0:
                    modules.append(nn.Dropout(p=dropout))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(self.hidden_dim, self.num_classes))
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        '''
            x: (batch_size x num_predictions x feature_dim)
            For each prediction, the hidden state from bert representing the execution
        '''
        if self.regex_features > 0:
            bert_features, regex_features = x
            batch_size, num_preds, _ = bert_features.shape
            x_projected = self.projection_layer(bert_features).reshape(batch_size, num_preds*self.projection_dim) #s (batch_size x (num_predictions*projection_dim))
            x_in = torch.cat([x_projected, regex_features],dim=-1)
            return self.model(x_in)
        else:
            x = x[0]
            batch_size, num_preds, _ = x.shape
            x_projected = self.projection_layer(x).reshape(batch_size, num_preds*self.projection_dim) #s (batch_size x (num_predictions*projection_dim))
            return self.model(x_projected)

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, 'weights.bin')
        torch.save(model_to_save.state_dict(), output_model_file)

