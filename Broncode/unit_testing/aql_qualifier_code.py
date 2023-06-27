"""
   Programm to qualify a batch of apples based on given aq-level and
   percentage of bad apples
   u_a = user_aql
"""


def batch_qualifier(u_a, perc_bad_apples_team):
    # """algorithm to help qualify the batch with apples based on percentage bad apples"""
    if perc_bad_apples_team <= u_a: 
        answer_team = "Class 1: Supermarket"
    elif u_a < perc_bad_apples_team < 6.5:
        answer_team = "Class 2: Apple sauce factory"
    elif 6.5 <= perc_bad_apples_team < 15:
        answer_team = "Class 3: Apple syrup factory"
    else:
        answer_team = "Class 4: Throw the apples away!"

    return answer_team


# verdict = batch_qualifier(4,3)    #"Class 1: Supermarket"
# verdict = batch_qualifier(4,6.4)  #"Class 2: Apple sauce factory"
# verdict = batch_qualifier(4,6.5)  #"Class 3: Apple syrup factory"
# verdict = batch_qualifier(4,6.6)  #"Class 3: Apple syrup factory"
# verdict = batch_qualifier(4,14.9) #"Class 3: Apple syrup factory"
# verdict = batch_qualifier(4,15)   #"Class 4: Throw the apples away!"
# verdict = batch_qualifier(4,15.1) #"Class 4: Throw the apples away!"
# print(verdict)

