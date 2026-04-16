import numpy as np

def simulate_needles(L, num_throws=1000000):

    # randomly generate the centers of the needles inside the [-1, 1]x[-1, 1] box
    xc = np.random.uniform(-1, 1, num_throws)
    yc = np.random.uniform(-1, 1, num_throws)
    
    # randomly generate the angles of the needles (0 to pi)
    phi = np.random.uniform(0, np.pi, num_throws)
    

    # sample 20 points along the length of each needle to check for intersections
    # if a needle crosses the curve, some points will be "inside" and some "outside"
    t = np.linspace(-L/2, L/2, 20)
    
    crosses = 0


    # batches for better performance
    batch_size = 100000
    for i in range(0, num_throws, batch_size):

        xc_b = xc[i:i+batch_size, np.newaxis]

        yc_b = yc[i:i+batch_size, np.newaxis]

        phi_b = phi[i:i+batch_size, np.newaxis]
        


        # calculate x, y coordinates for each needle
        x_pts = xc_b + t * np.cos(phi_b)
        y_pts = yc_b + t * np.sin(phi_b)
        


        # equation for the quadrifolium r = sin(2*theta) converted to cartesian is:
        # F(x, y) = (x^2 + y^2)^3 - 4(x^2)(y^2) = 0
        # if F < 0 the point is inside the curve but if F > 0 it's outside.
        r2 = x_pts**2 + y_pts**2
        F = (r2**3) - 4 * (x_pts**2) * (y_pts**2)
        

        # see if needle has points both F < 0 and F > 0
        has_pos = np.any(F > 0, axis=1)
        has_neg = np.any(F < 0, axis=1)
        
        # count needle if it crosses boundary
        crosses += np.sum(has_pos & has_neg)
        
    return crosses / num_throws



# run for each needle length
lengths = [1/10, 1/5, 1/4, 1/3]

throws = 1000000


print(f"Running Monte Carlo simulation with {throws:,} throws per length...\n")
print(f"{'Needle Length (L)':<20} | {'Crossing Probability'}")
print("-" * 45)

for L in lengths:

    prob = simulate_needles(L, throws)
   
    # formatting
    if L == 1/10: L_str = "1/10"

    elif L == 1/5: L_str = "1/5"

    elif L == 1/4: L_str = "1/4"

    else: L_str = "1/3"
    
    print(f"{L_str:<20} | {prob:.6f}")