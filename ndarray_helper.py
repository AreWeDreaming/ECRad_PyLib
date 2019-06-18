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

def ndarray_check_for_None(ndarray):
    any_None = False
    try:
        len(ndarray)
        for i, sub_ndarray in enumerate(ndarray):
            if(ndarray_check_for_None(ndarray[i])):
                return True
    except TypeError:
        any_None = ndarray is None
    return any_None