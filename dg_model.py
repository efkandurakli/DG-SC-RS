import torch
import torchvision
import torch.nn as nn
from typing import Any, Optional, Tuple
from torch.autograd import Function

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 0.1
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class DomainClassifier(nn.Module):
    def __init__(self, dim, num_domains):
        super().__init__()
        self.dim = dim

        self.gradient_reverse_layer = GradientReverseLayer()
        self.classifier = nn.Linear(dim, num_domains)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        x=self.gradient_reverse_layer(x)
        x=self.classifier(x)
        
        return self.softmax(x)

class DGModel(nn.Module):
    def __init__(self, model, weights, num_classes, num_domains=3):
        super().__init__()
        self.model =  torchvision.models.get_model(model, weights=weights, num_classes=num_classes)

        if self.training:
            self.domain_classifier = DomainClassifier(self.model.fc.in_features, num_domains)
            self.model.fc.register_forward_hook(self.store_fc_features)
    
    def store_fc_features(self, module, input, output):
        self.fc_features = input[0]


    def forward(self, x):
        if self.training:
            return self.model(x), self.domain_classifier(self.fc_features)
        
        return self.model(x)