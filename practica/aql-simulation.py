#----------------------------------------

import random as rd

#----------------------------------------

# Define the params
lotSize = 1_500
sampleSize = 125

# Level is defined as percentage
aql = 2.5 / 100

#----------------------------------------

# No pun intended!
numBadCups = round(aql * lotSize)
numGoodCups = lotSize - numBadCups

print(f"Number of bad plastic cups: {numBadCups}")
print(f"Number of good plastic cups: {numGoodCups}")

cupsBatch = []

# Append bad cups (encoded as 0)
for i in range(numBadCups):

    cupsBatch.append(0)

# Append good cups (encoded as 1)
for i in range(numGoodCups):

    cupsBatch.append(1)

#----------------------------------------

inspectionRounds = 10

results = []

for i in range(inspectionRounds):

    # Draw some random cups from the batch
    samples = rd.sample(cupsBatch, k = sampleSize)

    # Start fresh
    bads = 0

    # Count the number of bad cups
    for s in samples:

        # Bad cup?
        if s == 0:

            bads += 1

    # Store in results list
    results.append(bads)

#----------------------------------------

# Extract some statistics
avgBads = sum(results) / len(results)
maxBads = max(results)

print(f"The AVERAGE number of bad cups found is {avgBads}")
print(f"The MAXIMUM number of bad cups found is {maxBads}")

#----------------------------------------