"""
   Programm to test wether aql_qualifier / batch_qualifier works as intended
"""

from aql_qualifier_code import batch_qualifier
#---------------------------------------


def testBatch_Qualifier():
  
    assert batch_qualifier(4,3) == "Class 1: Supermarket","Check whether percentage is lower than u_a"
    assert batch_qualifier(4,6.4) == "Class 2: Apple sauce factory","Check the 6.5 limit."
    assert batch_qualifier(4,6.5) == "Class 3: Apple syrup factory","Check the 6.5 limit."
    assert batch_qualifier(4,6.6) == "Class 3: Apple syrup factory","Check the 6.5 limit."
    assert batch_qualifier(4,14.9) == "Class 3: Apple syrup factory","Check the 15 limit."
    assert batch_qualifier(4,15) == "Class 4: Throw the apples away!","Check the 15 limit."
    assert batch_qualifier(4,15.1) == "Class 4: Throw the apples away!","Check the 15 limit."

#---------------------------------------

def run_test():

    testBatch_Qualifier()

   
#---------------------------------------