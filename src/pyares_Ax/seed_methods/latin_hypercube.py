#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /src/pyares_bo/seed_methods/latin_hypercube.py
# Project: pyares-bo-planners
# Created Date: Friday, March 20th 2026, 1:13:44 pm
# Author(s): Arthur W. N. Sloan, Robert Waelder
# -----
# MIT License
# 
# Copyright (c) 2026 AFRL-ARES
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
###

import lhsmdu
import pandas as pd
import numpy as np
import sympy as sp
from time import time

class LHSMDU_Generator:
    def __init__(self,parameters:list[dict], N_points:int,rng_seed:int|None=None):
        self.parameters = parameters
        self.N_points = N_points
        self.N_params = len(parameters)
        if isinstance(rng_seed,int):
            self.rng_seed = rng_seed
            lhsmdu.setRandomSeed(rng_seed)
        else:
            self.rng_seed = int(time())
        self.unity_hypercube = lhsmdu.sample(self.N_params, self.N_points)
    def make_hypercube(self) -> pd.DataFrame:
        hypercube = pd.DataFrame()
        for i in range(self.N_params):
            if self.parameters[i]['type'] == 'range':
                param_min, param_max = self.parameters[i]['bounds']
                diff = param_max - param_min
                row = [diff * self.unity_hypercube[i, j] + param_min for j in range(np.shape(self.unity_hypercube)[1])]
                hypercube[self.parameters[i]['name']] = row
            elif self.parameters[i]['type'] == 'choice':
                values = self.parameters[i]['values']
                row = self.map_values(values,self.unity_hypercube[i])
                hypercube[self.parameters[i]['name']] = row
                
        
        return hypercube
    def make_constrained_hypercube(self, constraints:list[str])-> pd.DataFrame:
        N_valid = 0 
        mod = 0
        while N_valid != self.N_points:
            self.unity_hypercube = lhsmdu.sample(self.N_params, self.N_points+mod)
            hypercube = self.make_hypercube()
            mask = np.ones(len(hypercube),dtype=bool)
            for con in constraints:
                expr = sp.sympify(con) # Should probably figure out some input santiization in here
                symbols = list(expr.free_symbols)
                symbol_names = [symbol.name for symbol in symbols]
                func = sp.lambdify(symbols, expr, modules='numpy')
                args = [hypercube[name].to_numpy() for name in symbol_names]
                mask &= func(*args)
            
            const_hypercube = hypercube[mask].reset_index(drop=True)
            N_valid = len(const_hypercube)
            print(f"{N_valid}/{self.N_points}")
            if N_valid < self.N_points:
                mod += 1
            elif N_valid > self.N_points:
                mod -= 1
        
        hypercube = const_hypercube

        return hypercube
    @staticmethod
    def map_values(values,row):
        # The Hypercube produces values that are uniform in [0,1], which doesn't work for categorical variables,
        # Make a categorical sampling by binning the uniform distribuiton into a number of bins that corresponds
        # to the number of allowed values and then uses bin numbers to get level values
        
        N_levels = len(values)
        level_array = np.digitize(np.asarray(row).ravel(),np.arange(0,1,1/N_levels))
        output_row=np.asarray(values)[level_array-1]
        return output_row

if __name__ == '__main__':
    
    parameters = [{'name':'x1','bounds':[-5,10],'type':'range'},
                  {'name':'x2','bounds':[0,1],'type':'range'},
                  {'name':'x3','bounds':[5,20],'type':'range'},
                  {'name':'c1','type':'choice','values':['a','b','c','d']}]
    
    constraints = ['x1 + x2 < 9',
                   'x3 >= x1']
    cube_gen = LHSMDU_Generator(parameters,100,rng_seed=159753)
    hypercube = cube_gen.make_hypercube()

    c_cube = cube_gen.make_constrained_hypercube(constraints)
    
    c_cube.to_csv('test_cube.csv',index=False)