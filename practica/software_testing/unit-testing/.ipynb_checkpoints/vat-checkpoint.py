#---------------------------------------

# Base is 100%, VAT is 20% of base
def incluceVAT(base):

    return 1.2 * base

#---------------------------------------

# Total is 20% VAT + 100% base = 120%
def extractVAT(total):

    return 0.2 * (total / 1.2)

#---------------------------------------