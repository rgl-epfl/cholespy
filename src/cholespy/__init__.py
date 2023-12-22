from importlib import import_module
import_module('cholespy._cholespy_core')
del import_module

CHOLMOD_A = 0
CHOLMOD_LDLt = 1
CHOLMOD_LD = 2
CHOLMOD_DLt = 3
CHOLMOD_L = 4
CHOLMOD_Lt = 5
CHOLMOD_D = 6
CHOLMOD_P = 7
CHOLMOD_Pt = 8

