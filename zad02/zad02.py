import math
import random
import numpy as np
import matplotlib.pyplot as plt

v0 = 50
h = 100
g = 9.81

def sin(degrees):
    return math.sin(math.radians(degrees))

def cos(degrees):
    return math.cos(math.radians(degrees))

def showPlot(v, h, sin, cos, g):
    x = np.linspace(0, 340, 340)  # Generating 100 points between -2 and 2

    # Calculate y values (exponential function)
    y = -(g/(2*v**2*cos**2)) * x**2 + (sin/cos)*x + h

    y_masked = np.where(y >= 0, y, np.nan)

    # Create the plot
    plt.plot(x, y_masked)

    # Add labels and title
    plt.xlabel('Distance(m)')
    plt.ylabel('Height(m)')
    plt.title('Project Motion for the Trebuchet')

    # Add grid
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig('trajektoria.png')

    plt.show()

target = random.randint(50, 340)
attempts = 0
print('cel:', target)

while(True):
    alpha = int(input("Podaj kąt do strzału z Warwolfa: "))
    if alpha < 90:
        distance = round((v0 * sin(alpha) + math.sqrt(v0**2 * sin(alpha)**2 + 2*g*h)) * (v0 * cos(alpha)) / g)
        print(distance)
        if distance < target + 5 and distance > target - 5:
            print("Cel trafiony!")
            showPlot(v0, h, sin(alpha), cos(alpha), g)
            break
        else:
            print("Spróbuj jeszcze raz")
        attempts += 1
    else:
        print("Błędy kąt strzału, Kąt strzału musi byc z zakresu [-90, 90]")