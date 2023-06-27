#---------------------------------------

import sys
 
# Add reference to logging module
sys.path.append('./logger')

#---------------------------------------

import logger as LoggerModule

#---------------------------------------

import vat

#---------------------------------------

def testVAT():

    if vat.incluceVAT(100) != 120:

        logger.logFatal("incluceVAT test failed")
    
    if vat.extractVAT(120) != 20:

        logger.logFatal("extractVAT test failed")

#---------------------------------------

def testSort():

    unordered = [2, 9, 1, 6]

    ordered = sorted(unordered)

    assert ordered == [1, 2, 6, 9], f"Unordered list not properly sorted. Expected [1, 2, 6, 9] got {unordered}"

#---------------------------------------

def runTests():

    testVAT()

    testSort()

#---------------------------------------