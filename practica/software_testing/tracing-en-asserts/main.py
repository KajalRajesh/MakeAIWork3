#---------------------------------------

import sys
 
# Add reference to logging module
sys.path.append('./logger')

#---------------------------------------

import logger as LoggerModule

#---------------------------------------

def main():

    # Some dummy counter
    counter = 0

    # Will throw an exception when not true
    assert (counter == 0)

    logger = LoggerModule.Logger()

    logger.setLogLevel(LoggerModule.LEVEL_INFO)

    # Let's see what this does...
    logger.logInfo("Some info (1)")
    logger.logError("Some error (1)")

    # Some silly update
    counter += 2

    # Will throw an exception when not true
    assert (counter == 2)

    logger.setLogLevel(LoggerModule.LEVEL_ERROR)

    # And what this does...
    logger.logInfo("Some info (2)")
    logger.logError("Some error (2)")

#---------------------------------------

if __name__ == "__main__":

    main()

#---------------------------------------