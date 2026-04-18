import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

G = 9.8

class GeneralThreeBodyEnv:
    def __init__(self, target_orbit, masses, 
        macro_dt=1.0, micro_dt = .01, 
        initial_x = None, initial_v= None,
        max_fuel=10, radii=torch.tensor([1,1,1])):
        """
        Initializes the simulator.
        target_orbit_params: Collection of points and velocities in the orbit.
        macro_dt: The Agent Macro-step duration (Delta T).
        micro_dt: The Simulator Micro-step duration (Delta t)
        """
        self.dT = macro_dt
        self.dt = micro_dt
        self.target_orbit = target_orbit
        self.max_fuel = max_fuel
        self.w_1, self.w_2, self.w_3 = .5, .3, .2
        
        self.x = (initial_x if initial_x is not None 
            else torch.rand((3,3), dtype=torch.float64))
        self.v = (initial_v if initial_v is not None
            else torch.rand((3,3), dtype=torch.float64))
        
        self.m = masses

        self.m_10 = masses[1,2,0]
        self.m_20 = masses[2,0,1]

        self.radii = radii

    def reset(self):
        """
        Resets the universe to a random, slightly unstable initial configuration.
        Returns the initial graph state (nodes and edges).
        """
        # TODO: Initialize self.positions, self.velocities, self.masses
        
        return self._get_graph_state()

    def _get_graph_state(self):
        """
        Converts raw physics arrays into PyTorch tensors for the GNN.
        Returns node_features (pos, vel, mass) and edge_features (relative distances).
        """
        # TODO: Construct and return the graph dictionary or PyTorch Geometric Data object
        pass

    def _compute_derivs(self, x, v):
        x_10, x_20 = (x - x[:,[1,2,0]]), (x - x[:,[2,0,1]])
        v_10, v_20 = (v - v[:,[1,2,0]]), (v - v[:,[2,0,1]])

        x_10_norm, x_20_norm = torch.sqrt(torch.sum(x_10 * x_10, dim=0)), torch.sqrt(torch.sum(x_20 * x_20, dim=0))
        
        a_g = G * ((x_10 * (1 / torch.pow(x_10_norm, 3)) * self.m_10)
                + (x_20 * (1 / torch.pow(x_20_norm, 3)) * self.m_20))
        j = G * (((v_10 * (1 / torch.pow(x_10_norm, 3))) - 
                    (x_10 * (torch.sum(v_10 * x_10, dim=0)))
                    * (1 / torch.pow(x_10_norm, 5))) * self.m_10 +
                ((v_10 * (1 / torch.pow(x_10_norm, 3))) - 
                    (x_10 * (torch.sum(v_10 * x_10, dim=0)))
                    * (1 / torch.pow(x_10_norm, 5))) * self.m_10)
        
        return a_g, j
    def _sim_step(self, a_t):
        """
        The core physics engine. Uses the 4th-order Hermite scheme and adaptive 
        micro-steps (controlled by mu) to advance the simulation by macro_dt.
        agent_thrusts: Shape (3, 3), the constant acceleration applied by the agents.
        """
        
        K = self.dT // self.dt

        for _ in range(K):
            a_g_0, j_0 = self._compute_derivs(self, self.x, self.v)
            a_0 = a_g_0 + a_t

            x_p = self.x + self.v * self.dt + (1 / 2) * a_0 * (self.dt**2) + (1 / 6) * j_0 * (self.dt**3)
            v_p = self.v + a_0 * self.dt + (1 / 2) * j_0 * (self.dt**2)

            a_g_p, j_p = self._compute_derivs(self, x_p, v_p)
            a_p = a_g_p + a_t

            v_t = (1 / 2) * (a_0 + a_p) * self.dt + (1 / 12) * (j_0 - j_p) * (self.dt**2)
            self.x += (self.v + (1 / 2) * v_t) * self.dt + (1 / 12) * (a_0 - a_p) * (self.dt**2)
            self.v += v_t

    def check_collision(self):
        x_10 , x_20 = (self.x - self.x[:,[1,2,0]]), (self.x - self.x[:,[2,0,1]])
        x_10_norm, x_20_norm = (torch.sqrt(torch.sum(x_10 * x_10, dim=0)),
            torch.sqrt(torch.sum(x_20 * x_20, dim=0)))
        
        r_10, r_20 = ((self.radii + self.radii[[1,2,0]]),
            (self.radii + self.radii[[2,0,1]]))
        
        return torch.any(x_10_norm < r_10) or torch.any(x_20_norm < r_20)

    def step(self, a_t):
        """
        Executes one MARL Macro-step.
        actions: Continuous thrust vectors [Delta vx, Delta vy, Delta vz] for each body.
        """
        self._sim_step(a_t)
        
        R_orbit = None
        R_fuel = torch.log(torch.sum(a_t * a_t) / self.max_fuel)
        R_survive = -100 if self.check_collision else .1
        
        reward = self.w_1 * R_orbit + self.w_2 * R_fuel + self.w_3 * R_survive
        done = R_survive < 0
        
        return self._get_graph_state(), reward, done, {}

class PhysicsGNN(nn.Module):
    def __init__(self, node_in_dim=7, edge_in_dim=4, embed_dim=64):
        """
        Permutation-invariant Graph Neural Network.
        node_in_dim: 3(pos) + 3(vel) + 1(mass) = 7
        edge_in_dim: 3(rel_pos) + 1(distance) = 4
        """
        super(PhysicsGNN, self).__init__()
        
        # TODO: Define MLPs for Message Generation (phi_e)
        # TODO: Define MLPs for Node Update (phi_n)
        self.embed_dim = embed_dim

    def forward(self, nodes, edges):
        """
        Performs Message Passing.
        Returns a permutation-invariant embedding for each body.
        """
        # TODO: Implement message passing loop for K layers
        # Returns embeddings of shape (3 bodies, embed_dim)
        pass

class CelestialActor(nn.Module):
    def __init__(self, embed_dim=64, action_dim=3):
        """
        The decentralized 'Pilot' brain. 
        Because masses are different, we instantiate 3 of these (Star, Planet, Moon).
        """
        super(CelestialActor, self).__init__()
        
        # Simple MLP that takes the GNN embedding and outputs action distributions
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # For continuous PPO, we output the Mean and Log-Std of a Gaussian
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std_layer = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, body_embedding):
        """
        Takes ONLY its own embedding from the GNN.
        Returns the thrust probability distribution.
        """
        x = self.net(body_embedding)
        action_mean = torch.tanh(self.mean_layer(x)) # Tanh bounds thrust between -1 and 1
        action_std = torch.exp(self.log_std_layer)
        return action_mean, action_std

class CentralizedCritic(nn.Module):
    def __init__(self, embed_dim=64):
        """
        The 'God-view' Judge used ONLY during training (CTDE).
        """
        super(CentralizedCritic, self).__init__()
        
        # Takes the concatenated embeddings of ALL 3 bodies (3 * embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Outputs a single Value scalar V(s)
        )

    def forward(self, all_embeddings):
        return self.net(all_embeddings)

def train_active_orchestration():
    # 1. Initialize Environment
    env = GeneralThreeBodyEnv(target_orbit_params="figure_8")
    
    # 2. Initialize Neural Networks
    gnn = PhysicsGNN()
    
    # Heterogeneous Actors (One for each unique mass)
    actor_star = CelestialActor()
    actor_planet = CelestialActor()
    actor_moon = CelestialActor()
    actors = [actor_star, actor_planet, actor_moon]
    
    # Centralized Critic
    critic = CentralizedCritic()
    
    # TODO: Define PyTorch Optimizers for GNN, Actors, and Critic
    
    epochs = 1000
    for epoch in range(epochs):
        graph_state = env.reset()
        done = False
        
        while not done:
            # --- EXECUTION PHASE (Decentralized) ---
            with torch.no_grad():
                # 1. GNN processes the universe into 3 embeddings
                embeddings = gnn(graph_state['nodes'], graph_state['edges'])
                
                # 2. Each Actor looks ONLY at its own embedding to pick a thrust
                actions = []
                for i, actor in enumerate(actors):
                    mean, std = actor(embeddings[i])
                    # TODO: Sample action from Normal distribution(mean, std)
                    # actions.append(sampled_thrust)
            
            # 3. Step the environment
            next_graph_state, reward, done, info = env.step(actions)
            
            # --- TRAINING PHASE (Centralized - Usually done in batches via PPO) ---
            # TODO: Calculate Advantage using the Centralized Critic: A = Reward + gamma * V(next_s) - V(s)
            # TODO: Update Actor networks using PPO Clipped Objective
            # TODO: Update Critic network using Mean Squared Error loss against the actual returns
            # TODO: Update GNN gradients alongside the Actors/Critic
            
            graph_state = next_graph_state
            
        print(f"Epoch {epoch} complete. Reward: {reward}")

if __name__ == "__main__":
    # train_active_orchestration()
    print("Scaffold initialized.")