import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Constants
ALPHA_MAX = 90
TOLERANCE = 5

v0 = 50
h = 100
g = 9.81

# Function to convert degrees to radians
def radians(degrees):
    return math.radians(degrees)

# Function to simulate projectile motion and plot trajectory
def showPlot(v, h, sin_alpha, cos_alpha, g):
    x = np.linspace(0, 340, 340)

    y = -(g/(2*v**2*cos_alpha**2)) * x**2 + (sin_alpha/cos_alpha)*x + h
    y_masked = np.where(y >= 0, y, np.nan)

    plt.plot(x, y_masked)
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.title('Projectile Motion for the Trebuchet')
    plt.grid(True)
    plt.savefig('trajectory.png')
    plt.show()

# Main code
target = random.randint(50, 340)
attempts = 0
print('Target:', target)

while True:
    try:
        alpha = int(input("Enter launch angle (degrees) for the trebuchet: "))
        if abs(alpha) <= ALPHA_MAX:
            distance = round((v0 * math.sin(radians(alpha)) + math.sqrt(v0**2 * math.sin(radians(alpha))**2 + 2*g*h)) * (v0 * math.cos(radians(alpha))) / g)
            print("Distance:", distance, "m")
            if target - TOLERANCE <= distance <= target + TOLERANCE:
                print("Target hit!")
                showPlot(v0, h, math.sin(radians(alpha)), math.cos(radians(alpha)), g)
                break
            else:
                print("Try again.")
                attempts += 1
        else:
            print("Invalid angle. Angle must be in range [-90, 90].")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")