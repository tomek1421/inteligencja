import math

def fct_act(x):
    return 1 / (1 + pow(math.e, -x))

def forwardPass(wiek, waga, wzrost):
    hidden1 = -0.46122 * wiek + 0.97314 * waga + -0.39203 * wzrost + 0.80109
    hidden1_po_aktywacji = fct_act(hidden1)
    hidden2 = 0.78548 * wiek + 2.10584 * waga + -0.57847 * wzrost + 0.43529
    hidden2_po_aktywacji = fct_act(hidden2)
    output = -0.81546 * hidden1_po_aktywacji + 1.03775 * hidden2_po_aktywacji - 0.2368
    return output

print("forwardPass(23, 75, 176): ", forwardPass(23, 75, 176))
print("forwardPass(25, 67, 180)", forwardPass(25, 67, 180))