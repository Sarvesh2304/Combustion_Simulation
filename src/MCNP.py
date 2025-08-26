import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_particle_simulation(n_particles=1000, n_steps=100, dt=0.01):
    """
    Simulates N particles undergoing random motion in 2D space
    
    Parameters:
    -----------
    n_particles : int
        Number of particles to simulate
    n_steps : int 
        Number of time steps to simulate
    dt : float
        Time step size
        
    Returns:
    --------
    positions : array
        Array of particle positions at each time step
    """
    
    # Initialize particle positions
    positions = np.zeros((n_steps, n_particles, 2))
    positions[0] = np.random.randn(n_particles, 2)
    
    # Simulation parameters
    temperature = 300  # Kelvin
    kb = 1.380649e-23  # Boltzmann constant
    mass = 1e-26  # particle mass in kg
    
    # Simulate particle motion
    for step in range(1, n_steps):
        # Random thermal velocities
        velocities = np.sqrt(kb * temperature / mass) * np.random.randn(n_particles, 2)
        
        # Update positions using velocity verlet algorithm
        positions[step] = positions[step-1] + velocities * dt
        
    return positions

if __name__ == "__main__":
    # Run simulation
    n_particles = 1000
    n_steps = 100
    positions = monte_carlo_particle_simulation(n_particles, n_steps)

    # Visualize results
    plt.figure(figsize=(10, 10))
    plt.plot(positions[-1, :, 0], positions[-1, :, 1], 'b.', alpha=0.5)
    plt.title('Final Particle Positions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # Plot particle trajectories for a few particles
    plt.figure(figsize=(10, 10))
    for i in range(5):  # Plot first 5 particles
        plt.plot(positions[:, i, 0], positions[:, i, 1], '-', alpha=0.5, label=f'Particle {i+1}')
    plt.title('Particle Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
