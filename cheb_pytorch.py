from math import pi
import torch
from torch import nn


# TODO: incorporate batch processing with padded sequences


class ChebyshevApproximation(nn.Module):
    
    
    def __init__(self, deg):
        super(ChebyshevApproximation, self).__init__()
        
        self.deg = deg
        
        # corresponds to x_k where k indexes elements.
        k = torch.arange(deg).to(torch.float)
        self.nodes = torch.sort(torch.cos(pi * (k + 0.5) / deg))[0]
        
        # corresponds to (2-\delta_{0n})/N*T_n(x_k) where n indexes rows and k indexes columns.
        basis_normalization = (2 - (torch.arange(deg)==0).float()) / deg
        self.transposed_normalized_basis = basis_normalization * torch.cos(k * torch.acos(self.nodes.view(-1, 1)))
        
        
    def _interp_at_nodes_walk(self, x, y):
        
        interpolation = []
        m = len(x) - 1
        yT = y.t()  # transposed so that rows are indexed by x. makes indexing easier/cleaner
        
        if (self.nodes[0] < x[0]) or (self.nodes[-1] > x[-1]):
            raise Error('Cannot interpolate using input `x`.')
        
        i = 0
        
        for node in self.nodes:
            
            # linear walker along `x` for locations where all elements of `nodes` fits
            
            while x[i+1] < node:
                i += 1

            a, b = x[i], x[i+1]            
            node_interp = (node - a) / (b - a)
            interpolated_obs_vector = (1 - node_interp) * yT[i] + node_interp * yT[i+1]
            interpolation.append(interpolated_obs_vector)
        
        return torch.stack(interpolation, dim=1)  # stack so that each node indexes a column in the returned array (i.e., same format as original input y)

    
    def _interp_at_nodes_bsch(self, x, y):
        
        interpolation = []
        m = len(x) - 1
        yT = y.t()  # transposed so that rows are indexed by x. makes indexing easier/cleaner
        
        if (self.nodes[0] < x[0]) or (self.nodes[-1] > x[-1]):
            raise Error('Cannot interpolate using input `x`.')
        
        # `left` is out here on purpose. we assume that the input `x` and the self.nodes object are sorted, 
        # which means the left bound remains valid throughout and need not be reset.
        left = 0
        
        for node in self.nodes:
            
            # standard binary search for insertion points
            
            # `right`, however, must be reset for each search 
            # as we cannot easily derive an upper bound for the next node in general.
            right = m

            if x[left] == node:
                interpolation.append(yT[left])
                continue
            elif x[right] == node:
                interpolation.append(yT[right])
                left = right
                right = left + 1
                continue

            mid = int(0.5 * (left + right))

            while mid != left:

                if node == x[mid]:
                    left = mid
                    right = mid + 1
                elif node < x[mid]:
                    right = mid
                elif node > x[mid]:
                    left = mid

                mid = int(0.5 * (left + right))

            a, b = x[left], x[right]
            node_interp = (node - a) / (b - a)
            interpolated_obs_vector = (1 - node_interp) * yT[left] + node_interp * yT[right]
            interpolation.append(interpolated_obs_vector)
        
        return torch.stack(interpolation, dim=1)  # stack so that each node indexes a column in the returned array (i.e., same format as original input y)
    
    
    def forward(self, x, y, method='bsch'):
        if method == 'bsch':
            y_at_nodes = self._interp_at_nodes_bsch(x, y)
        elif method == 'walk':
            y_at_nodes = self._interp_at_nodes_walk(x, y)
        else:
            raise Error('Argument `method` is unrecognized')
        return (y_at_nodes @ self.transposed_normalized_basis).flatten()
    
    
    
class InverseChebyshevApproximation(nn.Module):
    
    
    def __init__(self, deg):
        super(InverseChebyshevApproximation, self).__init__()
        
        self.deg = deg
    
    
    def forward(self, x, a):
        
        k = torch.arange(self.deg).to(torch.float)
        transposed_basis = torch.cos(k * torch.acos(x.view(-1, 1)))
        
        coefficients = a.reshape(-1, self.deg)
        
        # This reshape/broadcast witchcraft may be opaque, but suffice it to say, it works, and it's probably as fast as we'll ever get it.
        return (coefficients[:,None,:] * transposed_basis[None,:,:]).sum(dim=2)
    