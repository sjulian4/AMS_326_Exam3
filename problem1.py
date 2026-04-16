import numpy as np
import matplotlib.pyplot as plt

# h = 0.01
# 

# the ODE y' = f(x, y)
def f(x, y):
    return -(1/81) * (x**3) * (y**2) + np.sin(x**2) * np.cos(y**3)


# parameters
x0, y0 = 0.0, 0.0
x_end = 5.0
h = 0.01


# x values
x = np.arange(x0, x_end + h, h)
n = len(x)


# for y values for each method
y_euler = np.zeros(n)
y_heun = np.zeros(n)
y_rk4 = np.zeros(n)


# initial
y_euler[0] = y0
y_heun[0] = y0
y_rk4[0] = y0


# integration loop
for i in range(n - 1):

    xi = x[i]
    
    # 1. Forward Euler Method
    y_euler[i+1] = y_euler[i] + h * f(xi, y_euler[i])
    

    # 2. Heun's Method
    k1_heun = f(xi, y_heun[i])
    k2_heun = f(xi + h, y_heun[i] + h * k1_heun)
    y_heun[i+1] = y_heun[i] + (h / 2) * (k1_heun + k2_heun)
    


    # 3. 4th-Order Runge-Kutta (RK4) Method
    k1_rk = f(xi, y_rk4[i])
    k2_rk = f(xi + h/2, y_rk4[i] + h * k1_rk / 2)
    k3_rk = f(xi + h/2, y_rk4[i] + h * k2_rk / 2)
    k4_rk = f(xi + h, y_rk4[i] + h * k3_rk)
    y_rk4[i+1] = y_rk4[i] + (h / 6) * (k1_rk + 2*k2_rk + 2*k3_rk + k4_rk)



# plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y_rk4, 'k-', linewidth=2, label="4th-Order RK (RK4)")
plt.plot(x, y_heun, 'b--', linewidth=1.5, label="Heun's Method")
plt.plot(x, y_euler, 'r-.', linewidth=1.5, label="Forward Euler")

plt.title("Numerical Solutions of $y' = -\\frac{1}{81}x^3y^2 + \sin(x^2)\cos(y^3)$ with $h=0.01$", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y(x)", fontsize=12)
plt.legend(loc="best", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)


plt.savefig("ODE_solutions_plot.png", dpi=300, bbox_inches="tight")

plt.show()