"""
09-27-2023

Code for implementing Kohen's Kappa that compares the similarities between two
sets of labels for the same thing.
"""

import numpy as np

def cohens_kappa(labels1, labels2):
    # calculate the probability that the two labels agree with each other
    p0 = np.mean(labels1 == labels2)

    # calculate the probability that the two labels agree with each other
    # according to random chance
    labels1_pyes = np.mean(labels1)
    labels2_pyes = np.mean(labels2)
    labels1_pno = 1 - labels1_pyes
    labels2_pno = 1 - labels2_pyes
    pe = labels1_pyes*labels2_pyes + labels1_pno*labels2_pno

    return (p0-pe) / (1-pe)