import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MFGLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        factory_kwargs = {
            "device": kwargs.get("device", None),
            "dtype": kwargs.get("dtype", None),
        }

        # log variance for the weight distribution
        # variational distribution for the weight is N(self.weight, exp(self.weight_logvar))
        self.weight_logvar = nn.Parameter(
            torch.empty((self.out_features, self.in_features), **factory_kwargs)
        )
        nn.init.constant_(self.weight_logvar, -5.0)

        # log variance for the bias distribution
        # variational distribution for the bias is N(self.bias, exp(self.bias_logvar))
        if self.bias is not None:
            self.bias_logvar = nn.Parameter(
                torch.empty(self.out_features, **factory_kwargs)
            )
            nn.init.constant_(self.bias_logvar, -5.0)

    def forward(self, input, num_samples=1):

        if self.training:                            
            
            # forward pass via local reparameterization trick
            # fill in the blank
            #####################################################################################

            #####################################################################################

        else:
            weight_mean = self.weight
            weight_std = torch.exp(0.5 * self.weight_logvar)
            if self.bias is not None:
                bias_mean = self.bias
                bias_std = torch.exp(0.5 * self.bias_logvar)

            if num_samples > 1:
                weight_mean = torch.stack([weight_mean] * num_samples)
                weight_std = torch.stack([weight_std] * num_samples)
                if self.bias is not None:
                    bias_mean = torch.stack([bias_mean] * num_samples)
                    bias_std = torch.stack([bias_std] * num_samples)

                if input.ndim == 2:
                    input = torch.stack([input] * num_samples)

            weight = weight_mean + weight_std * torch.randn_like(weight_mean)
            bias = (
                0.0
                if self.bias is None
                else (bias_mean + bias_std * torch.randn_like(bias_mean)).unsqueeze(-2)
            )

            return input @ weight.transpose(-2, -1) + bias

    def KLD(self, prior_scale):
        prior_std = 1.0 / math.sqrt(prior_scale)

        # compute KL[N(mu, sigma^2) || N(0, prior_std^2)]
        def kld_(mu, sigma):
            
            # fill in the blank
            #####################################################################################

            #####################################################################################

        weight_kld = kld_(self.weight, torch.exp(0.5 * self.weight_logvar))
        bias_kld = (
            0.0
            if self.bias is None
            else kld_(self.bias, torch.exp(0.5 * self.bias_logvar))
        )

        return weight_kld + bias_kld


class MFGMLP(nn.Module):
    def __init__(self, num_inputs, num_hiddens, num_outputs, num_layers):
        super().__init__()
        self.input_layer = MFGLinear(num_inputs, num_hiddens)
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.append(MFGLinear(num_hiddens, num_hiddens))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.output_layer = MFGLinear(num_hiddens, num_outputs)

    def forward(self, x, num_samples=1):
        x = F.silu(self.input_layer(x, num_samples=num_samples))
        for layer in self.hidden_layers:
            x = F.silu(layer(x, num_samples=num_samples))
        x = F.log_softmax(self.output_layer(x, num_samples=num_samples), -1)
        return x

    def KLD(self, prior_scale):
        kld = self.input_layer.KLD(prior_scale)
        for layer in self.hidden_layers:
            kld += layer.KLD(prior_scale)
        kld += self.output_layer.KLD(prior_scale)
        return kld
