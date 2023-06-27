#! /usr/bin/python

"""
   Programm to run. If the test fails, an AsserError will appear and 
   batch_qualifier() won't start.
"""
#---------------------------------------

from aql_qualifier_code import batch_qualifier
from aq_test import testBatch_Qualifier
from aq_test import run_test

#---------------------------------------

def main():

    testBatch_Qualifier()

    verdict = batch_qualifier(4,3)   #"Class 1: Supermarket"
    print(verdict)
    # verdict = batch_qualifier(4,6.4)  #"Class 2: Apple sauce factory"
    # verdict = batch_qualifier(4,6.5)  #"Class 3: Apple syrup factory"
    # verdict = batch_qualifier(4,6.6)  #"Class 3: Apple syrup factory"
    # verdict = batch_qualifier(4,14.9) #"Class 3: Apple syrup factory"
    # verdict = batch_qualifier(4,15)   #"Class 4: Throw the apples away!"
    # verdict = batch_qualifier(4,15.1) #"Class 4: Throw the apples away!"
    
#---------------------------------------

if __name__ == "__main__":

    main()

#---------------------------------------