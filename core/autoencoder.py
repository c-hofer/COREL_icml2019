"""autoencoder.py

Author(s): chofer, rkwitt (2018)
"""

import torch
import torch.nn as nn
import core.mynn as mynn


class DCGEncDec(nn.Module):
    """
    Implementation of a convolutional autoencoder with a DCGAN-style
    encoder (disc. in DCGAN) and decoder. 

    Args:
        filter_config: list 
            Number of channels input to each conv. layer. The length of
            the list determines the number of layers. The filters in 
            each layer are 3x3 spatially with a step-size of two.

        input_config: list
            List of input img. dimensions in the form channel x width x height.

        latent_config: dict
            Specification of the latent space geometry. 

            Keys: 

                n_branchces: int 
                    number of independent branches in the latent space - Needs
                    to be a divisor of the dimensionality of the last conv.  
                    layer when flattened.

                out_features_branch: int
                    number of output features for each independent branch. 

        Example::
            m = DCGEncDec(
                filter_config = [3,16,32,64], 
                input_config  = [3,32,32],
                latent_config = {n_branches: 16, 'out_features_branch': 10}
            )
            This creates an autoencoder for input images of size 3x32x32 with
            3 input channels (obviously) and three conv. layers with 16,32 and
            64 filters (each followed by leaky ReLU activations). The decoder 
            mirrors this architecture with convolutional transpose filters. 

            In the latent space, this model has 16 branches
            that output 10 dimensional features. For decoding, these features
            are concatenated. In this particular example, flattening the output 
            of the last conv. layer results in a 1024-dim. representation.
    """
    def __init__(self, *ignore,
                 filter_config=[3,16,32,64], 
                 input_config=[3,32,32], 
                 latent_config={'n_branches': 1, 'out_features_branch': 10}):
        super().__init__()
        
        assert len(ignore) == 0, "Keyword args only!"

        self.n_branches = latent_config['n_branches']
        self.out_features_branch = latent_config['out_features_branch']
        
        assert(filter_config[0]==input_config[0])
        
        self.enc = []
        for i in range(len(filter_config)-1):
            self.enc.append(
                nn.Conv2d(in_channels  = filter_config[i], 
                          out_channels = filter_config[i+1], 
                          kernel_size  = 3, 
                          stride       = 2, 
                          padding      = 1, 
                          bias         = True)
            )
            
            self.enc.append(nn.LeakyReLU())
            
        self.enc_conv = nn.Sequential(*self.enc)

        # Compute the required size of the linear layer following the last conv. stage
        enc_dim = torch.tensor(list(self.enc_conv(torch.randn(10,*input_config)).size()[1:]))
        assert enc_dim.prod() % self.n_branches == 0
        
        # Make sure we have independent branches - this effectively multiplies the 
        # weight matrix of the linear layer by a mask which ensures this property.
        self.enc_fc = mynn.IndependentBranchesLinear(
            enc_dim.prod(), 
            self.out_features_branch,
            self.n_branches
        )
        
        # Create a linear view
        self.enc = nn.Sequential(
            self.enc_conv, 
            mynn.LinearView(),
            self.enc_fc
        )
        
        # Unfold the independent linear branches 
        self.dec_fc = mynn.IndependentBranchesLinear(
            self.latent_dim, 
            int(enc_dim.prod()/self.n_branches), 
            self.n_branches
        )
    
        self.dec_convt = []
        self.dec_convt.append(mynn.View(tuple([-1] + list(enc_dim))))
        
        reversed_filter_config = list(reversed(filter_config))
        for i in range(len(reversed_filter_config)-1):
            self.dec_convt.append(
                nn.ConvTranspose2d(
                    in_channels    = reversed_filter_config[i],
                    out_channels   = reversed_filter_config[i+1], 
                    kernel_size    = 3,
                    stride         = 2, 
                    padding        = 1, 
                    output_padding = 1)
            )
            self.dec_convt.append(nn.ReLU())

        # remove last ReLU
        self.dec_convt = self.dec_convt[:-1]
        
        # At the moment we do not use a Sigmoid output
        self.dec_convt = nn.Sequential(*self.dec_convt)
        self.dec = nn.Sequential(self.dec_fc,
                                 self.dec_convt)
                
    
    def forward(self, input):
        z = self.enc(input)
        x = self.dec(z)
        return x,z
    
    @property
    def latent_dim(self):
        return self.n_branches*self.out_features_branch

