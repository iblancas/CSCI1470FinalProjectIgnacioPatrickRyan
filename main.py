import torch
from engine import PhysicsEngine
from model.GNN import PhysicsGNN
from model.actor import CelestialActor
from model.critic import CentralizedCritic

def train_active_orchestration():
    # Initialize Environment
    engine = PhysicsEngine()
    
    # 2. Initialize Neural Networks
    gnn = PhysicsGNN()
    
    # Actors for each of the bodies
    actors = [CelestialActor() for _ in range(3)]
    
    # Centralized Critic
    critic = CentralizedCritic()
    
    # TODO: Define PyTorch Optimizers for GNN, Actors, and Critic
    
    epochs = 1000
    for epoch in range(epochs):
        graph_0 = engine.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                h = gnn(graph_0["nodes"], graph_0["edges"])
                
                actions = torch.zeros((3,3))
                for i, actor in enumerate(actors):
                    mean, std = actor(h[i])

                    a_i = torch.normal(mean, std)
                    actions[i] = a_i
            
            graph_1, reward, done, info = engine.step(actions)
            
            # --- TRAINING PHASE (Centralized - Usually done in batches via PPO) ---
            # TODO: Calculate Advantage using the Centralized Critic: A = Reward + gamma * V(next_s) - V(s)
            h_next = gnn(graph_1["nodes"], graph_1["edges"])
            
            # TODO: Update Actor networks using PPO Clipped Objective
            # TODO: Update Critic network using Mean Squared Error loss against the actual returns
            # TODO: Update GNN gradients alongside the Actors/Critic
            
            graph_0 = graph_1
            
        print(f"Epoch {epoch} complete. Reward: {reward}")

if __name__ == "__main__":
    # train_active_orchestration()
    print("Scaffold initialized.")