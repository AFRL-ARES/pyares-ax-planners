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

def map_values(values,row):
    # The Hypercube produces values that are uniform in [0,1], which doesn't work for categorical variables,
    # Make a categorical sampling by binning the uniform distribuiton into a number of bins that corresponds
    # to the number of allowed values and then uses bin numbers to get level values
    
    N_levels = len(values)
    level_array = np.digitize(np.asarray(row).ravel(),np.arange(0,1,1/N_levels))
    output_row=np.asarray(values)[level_array-1]
    return output_row
    
    
 
def make_hypercube(parameters:list[dict], N_points:int,rng_seed:int|None=None) -> pd.DataFrame:
    if isinstance(rng_seed,int):
        lhsmdu.setRandomSeed(rng_seed)
    N_params = len(parameters)
    # Generating points when all parameters are continuous ranges is easy enough
    # for categorical variables we will check how many cateogical levels are allowed, convert those to intergers and then convert the
    # uniformly distributed points into an integer depending on the range 
    # Here N points is the total number of points requested, so we need to divide by the 
    unity_hypercube = lhsmdu.sample(N_params, N_points)
    hypercube = pd.DataFrame()
    for i in range(N_params):
        if parameters[i]['type'] == 'range':
            param_min, param_max = parameters[i]['bounds']
            diff = param_max - param_min
            row = [diff *unity_hypercube[i, j] + param_min for j in range(N_points)]
            hypercube[parameters[i]['name']] = row
        elif parameters[i]['type'] == 'choice':
            values = parameters[i]['values']
            row = map_values(values,unity_hypercube[i])
            hypercube[parameters[i]['name']] = row
            
    
    return hypercube

def make_constrained_hypercube(parameters:list[dict], N_points:int, constraints:list[str], rng_seed:int|None=None)-> pd.DataFrame:
    # Some process spaces may be constrained, In this naive method, ponts are discareded if they violate a constraint, 
    # and the hypercube is resampled at increasing resolutions until enough valid points are found 
    N_valid = 0 
    mod = 0

    while N_valid != N_points:
        
        hypercube = make_hypercube(parameters,N_points+mod,rng_seed)
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
        print(N_valid)
        mod += 1
    
    hypercube = const_hypercube

    return hypercube

if __name__ == '__main__':
    
    parameters = [{'name':'x1','bounds':[-5,10],'type':'range'},
                  {'name':'x2','bounds':[0,1],'type':'range'},
                  {'name':'x3','bounds':[5,20],'type':'range'},
                  {'name':'c1','type':'choice','values':['a','b','c','d']}]
    
    constraints = ['x1 + x2 < 9',
                   'x3 >= x1']
    
    hypercube = make_hypercube(parameters,50,rng_seed=159753)

    c_cube = make_constrained_hypercube(parameters,50,constraints,rng_seed=159753)
    
    c_cube.to_csv('test_cube.csv',index=False)