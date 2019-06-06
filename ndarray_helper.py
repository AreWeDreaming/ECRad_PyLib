'''
Created on 04.06.2019

@author: sdenk
'''

def ndarray_math_operation(ndarray, operation):
    try:
        ndarray = operation(ndarray)
    except AttributeError:
        for i, sub_ndarray in enumerate(ndarray):
            ndarray[i]= ndarray_math_operation(ndarray[i], operation)
    return ndarray