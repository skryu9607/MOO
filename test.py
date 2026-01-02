import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_piecewise_linear_surface():
    # 1. Setup the weight space (w1, w2)
    w1 = np.linspace(0, 1, 20)
    w2 = np.linspace(0, 1, 20)
    W1, W2 = np.meshgrid(w1, w2)
    
    # 2. Define "Robot Plans" as linear planes
    # Plan A: Good for low w1, low w2
    # Plan B: Good for high w1
    # Plan C: Good for high w2
    # The optimal cost u(w) is the MINIMUM of these costs (concave)
    # (Note: In maximization problems it's max, here we visualize the 'tent' structure)
    
    # Let's create a "Dome/Tent" structure (Concave)
    # Plane 1: w1 + w2
    P1 = 0.5 * W1 + 0.5 * W2 + 0.2
    # Plane 2: Tilt towards W1
    P2 = 0.8 * W1 + 0.2 * W2 
    # Plane 3: Tilt towards W2
    P3 = 0.2 * W1 + 0.8 * W2 
    
    # The function u(w) is the pointwise MINIMUM of these planes 
    # (Creating the "creases" or "facets")
    Z = np.minimum(np.minimum(P1, P2), P3)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 3. Plot the "Tent" Surface
    surf = ax.plot_surface(W1, W2, Z, alpha=0.6, cmap='viridis', edgecolor='none')
    
    # 4. Highlight points on ONE Facet (The Hyperplane)
    # Let's pick points where P2 is the minimum (The "Plan B" region)
    # We will pick 5 random points that lie on this specific flat sheet
    
    # Create points that satisfy Z = P2 exactly
    mask = (P2 < P1) & (P2 < P3)
    
    # Manually pick a few points on this plane to show they are co-planar
    p_w1 = np.array([0.7, 0.8, 0.9, 0.75, 0.85])
    p_w2 = np.array([0.1, 0.1, 0.05, 0.15, 0.0])
    p_z  = 0.8 * p_w1 + 0.2 * p_w2 # Using the equation of Plane 2
    
    ax.scatter(p_w1, p_w2, p_z, color='red', s=100, label='Weights on the same Hyperplane')

    # Draw the wireframe to emphasize the flatness
    ax.plot_wireframe(W1, W2, Z, color='black', alpha=0.3, rstride=5, cstride=5)

    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Optimal Cost u(w)')
    ax.set_title('Piecewise Linear Cost Function ("The Tent")\nRed dots = Many points on ONE hyperplane')
    ax.legend()
    
    plt.show()

draw_piecewise_linear_surface()
