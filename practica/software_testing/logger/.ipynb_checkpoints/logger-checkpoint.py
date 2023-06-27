#---------------------------------------

# Constants
LEVEL_INFO = 3
LEVEL_WARN = 2
LEVEL_ERROR = 1
LEVEL_FATAL = 0

#---------------------------------------

class Logger:

    #---------------------------------------

    def __init__(self):

        # Default level
        self.logLevel = LEVEL_INFO

    #---------------------------------------

    def setLogLevel(self, level):

        self.logLevel = level

    #---------------------------------------

    def logInfo(self, msg):

        if self.logLevel >= LEVEL_INFO:

            print("INFO: ", msg)

    #---------------------------------------

    def logWarn(self, msg):

        if self.logLevel >= LEVEL_WARN:

            print("WARN: ", msg)

    #---------------------------------------

    def logError(self, msg):

        if self.logLevel >= LEVEL_ERROR:

            print("ERROR: ", msg)

    #---------------------------------------

    def logFatal(self, msg):

        if self.logLevel >= LEVEL_FATAL:

            print("FATAL: ", msg)

    #---------------------------------------

#---------------------------------------