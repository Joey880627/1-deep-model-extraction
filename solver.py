import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space
from model import Model
from utils import *

class Solver:
    def __init__(self, model):
        self.input_dim, self.hidden_dim = model.input_dim, model.hidden_dim
        self.model = model
        self.critical_points = []
        self.first_layer_weights = []
        self.first_layer_bias = []
        self.last_layer_weights = []
        self.last_layer_bias = []
        self.steal_model = None
        self.d = 1e-6
    
    def solve(self):
        print("-- Start attacking")
        # 1. Find critical points and use critical points to find weights with relative signs of the first layer
        print("---- Start finding critical points")
        iteration = 0
        while len(self.first_layer_weights) < self.hidden_dim:
            print(f"------ Iteration {iteration}, {len(self.critical_points)} critical points found")
            iteration += 1
            self._find_critical_points(start=10*np.random.randn(self.input_dim), end=10*np.random.randn(self.input_dim))
        print(f"---- Finished, {len(self.critical_points)} critical points found")
        self.critical_points = np.array(self.critical_points)
        self.first_layer_weights = np.array(self.first_layer_weights)
        self.first_layer_bias = np.array(self.first_layer_bias)
        
        # 2. Find absolute neuron signs
        print("---- Start calculating sign of each neuron")
        self._find_neuron_sign()
        print("---- Finish calculating sign of each neuron")
        #   2.1 Use normal method to find weights and bias of the output layer
        #   2.2 Validate the results by computing error of the linear regression
        print("---- Start calculating weights of the last layer")
        self._find_last_layer()
        steal_weights = [self.first_layer_weights.T, self.last_layer_weights]
        steal_bias = [self.first_layer_bias, self.last_layer_bias]
        self.steal_model = Model(self.input_dim, self.hidden_dim, steal_weights, steal_bias)
        
        print("---- Finish calculating weights of the last layer")
        print("-- Finish attacking\n")
        
        return self.steal_model
        
    def _input_generator(self, length):
        return np.random.randn(length, self.input_dim)
        
    def _find_critical_points(self, start, end):
        # Paper: 6.3
        # Perform binary search to find critical points
        # Terminate condition (start==end or diff_start==diff_end)
        if norm(start - end) < 1e-6: # start==end
            return
        diff_start = differential(self.model, start, self.d)
        diff_end = differential(self.model, end, self.d)
        if norm(diff_end - diff_start) < 1e-6: # diff_start==diff_end
            return
        
        mid = start + (end - start) / 2 # mid point
        second_diff_mid = second_differential(self.model, mid, self.d) # Second derivative of mid point
        
        # If second derivative is not zero -> critical point found
        if np.all(np.abs(second_diff_mid) > 1e3): # Second derivative is not zero
            diff_mid_plus = differential(self.model, mid, 1e-2)
            diff_mid_minus = differential(self.model, mid, -1e-2)
            
            weight = np.abs(diff_mid_plus - diff_mid_minus)
            
            # Append this weight to self.unique_weights if the weight hasn't be found yet
            if (not self._has_weight(weight, self.first_layer_weights)) and len(self.first_layer_weights) < self.hidden_dim:
                self.critical_points.append(mid)
                sign = self._find_input_sign(self.model, mid) # Start finding relative sign of the first layer
                weight *= sign
                self.first_layer_weights.append(weight)
                # Add bias
                bias = -(weight @ mid)
                self.first_layer_bias.append(bias)
            return
        
        diff_mid = differential(self.model, mid, self.d)
        self._find_critical_points(start, mid)
        self._find_critical_points(mid, end)
        return
        
    def _has_weight(self, new_weight, weights):
        for old_weight in weights:
            if norm(np.abs(old_weight) - np.abs(new_weight)) < 1e-3:
                return True
        return False
        
    def _find_input_sign(self, function, x, d=1e-6):
        # Paper: 6.4.2
        input_dim = x.shape[0]
        sign_second_differential = 1 if second_differential(self.model, x)[0] > 0 else -1
        sign = [1]
        
        # We compute d2f/dx_1*dx_k for each i when 1 < k < input_dim
        for k in range(1, input_dim):
            d_pp = np.zeros(x.shape)
            d_pp[0] = d
            d_pp[k] = d
            d_pm = np.zeros(x.shape)
            d_pm[0] = d
            d_pm[k] = -d
            d_mp = np.zeros(x.shape)
            d_mp[0] = -d
            d_mp[k] = d
            d_mm = np.zeros(x.shape)
            d_mm[0] = -d
            d_mm[k] = -d
            pp = function(x + d_pp)
            pm = function(x + d_pm)
            mp = function(x + d_mp)
            mm = function(x + d_mm)
            result = (pp-pm-mp+mm) / (4 * d**2)
            # The sign of dim 0 and dim k are different if the second partial derivative has negative impact
            sign.append( (1 if result>0 else -1) * sign_second_differential )
        return np.array(sign)
        
    def _find_neuron_sign(self):
        # Paper 6.4.1
        A = np.concatenate([self.first_layer_weights, np.expand_dims(self.first_layer_bias, 1)], 1)
        z = null_space(A)[:, 0]
        z /= z[-1]
        z = z[:-1]
        
        
        for i in range(self.hidden_dim):
            y = np.zeros((self.hidden_dim))
            y[i] = 1
            
            v = least_square(A, y)
            v /= v[-1]
            
            sign_A_dot_v = 1 if (A @ v)[i] > 0 else -1
            v = v[:-1]
            z_output = self.model(z)
            z_plus_v_output = self.model(z + v)
            z_minus_v_output = self.model(z - v)
            sign_output = 1 if abs(z_plus_v_output - z_output) > abs(z_minus_v_output - z_output) else -1
            if (sign_output != sign_A_dot_v):
                self.first_layer_weights[i] *= -1
                self.first_layer_bias[i] *= -1
                
    def _find_last_layer(self):
        # Paper 6.6
        # Using normal equation to find the weights of the last layer
        random_input = self._input_generator(100)
        X = []
        Y = []
        for x in random_input:
            hidden_representation = relu(self.first_layer_weights @ x + self.first_layer_bias)
            oracle = self.model(x)
            X.append(np.append(hidden_representation, 1))
            Y.append(oracle)
        X = np.array(X)
        Y = np.array(Y)
        w_b = normal_equation(X, Y)
        self.last_layer_weights = w_b[:-1]
        self.last_layer_bias = w_b[-1:]