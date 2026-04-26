import itertools
import torch
import torch.nn.functional as F
from engine import PhysicsEngine
from model.GNN import PhysicsGNN
from model.actor import CelestialActor
from model.critic import CentralizedCritic
from torch.distributions import Normal

GAMMA=.2
GAE_LAMBDA=.3
PPO_CLIP = .001
C1 = 1
C2 = 1

engine = PhysicsEngine(
    (torch.ones((10, 3, 3), dtype=torch.float64),
     torch.ones((10, 3, 3), dtype=torch.float64)),
     torch.tensor([1, 1, 1]), 1, .0001)
gnn = PhysicsGNN()
actors = [CelestialActor() for _ in range(3)]
critic = CentralizedCritic()

params = itertools.chain(
    gnn.parameters(),
    actors[0].parameters(),
    actors[1].parameters(),
    actors[2].parameters(),
    critic.parameters()
)

optimizer = torch.optim.Adam(params, lr=3e-4)

def train_active_orchestration(engine, gnn, actors, critic, optimizer,
    epochs=15, k_epochs=15, batch_size=64, old_data_size=1000):
    for epoch in range(epochs):
        graph_0 = engine.reset()
        done = False

        old_data = {
            "nodes": [],
            "edges": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "is_done": [],
        }
        
        for i in range(old_data_size):
            print(i)
            with torch.no_grad():
                h = gnn(graph_0["nodes"].unsqueeze(0),
                    graph_0["edges"].unsqueeze(0))
                
                actions = torch.zeros((3,3))
                log_prob = 0
                for i, actor in enumerate(actors):
                    mean, std = actor(h[:,i,:])
                    distrib = Normal(mean, std)

                    a_i = distrib.sample()
                    actions[i] = a_i
                    log_prob += distrib.log_prob(a_i).sum(dim=-1)
                value = critic(torch.flatten(h, start_dim=-2)).squeeze(-1)
            graph_1, reward, done, info = engine.step(actions)
            
            old_data["nodes"].append(graph_0["nodes"])
            old_data["edges"].append(graph_0["edges"])
            old_data["actions"].append(actions)
            old_data["log_probs"].append(log_prob)
            old_data["rewards"].append(reward)
            old_data["values"].append(value)
            old_data["is_done"].append(0 if done else 1)

            graph_0 = graph_1         

            # If we detect a collision we exit the simulation
            if done: break
        
        # Number of generated time_steps
        B = len(old_data["nodes"])
        
        # Compute advantages        
        advantages= torch.zeros((B,))
        returns = []
        last_gae_lam = 0

        for i in range(B):
            t = B - 1 - i
            if i == 0:
                with torch.no_grad():
                    value_1 = critic(torch.flatten(
                        gnn(graph_0["nodes"].unsqueeze(0),
                            graph_0["edges"].unsqueeze(0)),
                        start_dim=-2)).squeeze(-1)
            else:
                value_1 = old_data["values"][t + 1]
            delta = old_data['rewards'][t] + GAMMA * value_1 * old_data["is_done"][t] - old_data['values'][t]
            last_gae_lam = delta + GAMMA * GAE_LAMBDA * old_data["is_done"][t] * last_gae_lam
            advantages[t] = last_gae_lam

            returns.insert(0, last_gae_lam + old_data["values"][t])
        returns = torch.tensor(returns).flatten()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        advantages = advantages.flatten()
        print(advantages.shape)

        # Iterate policy changes over rollout data
        old_data["nodes"] = torch.stack(old_data["nodes"])
        old_data["edges"] = torch.stack(old_data["edges"])
        old_data["actions"] = torch.stack(old_data["actions"])
        old_data["log_probs"] = torch.tensor(old_data["log_probs"])
        old_data["rewards"] = torch.tensor(old_data["rewards"])
        old_data["values"] = torch.tensor(old_data["values"])

        print("-"*10 + "Rollout data computed" + "-"*10)

        for e in range(k_epochs):
            idx = torch.randperm(B)
            # Mini batches on the old data
            for b_i in range(0, B, batch_size):
                print(b_i)
                b_nodes = old_data["nodes"][idx[b_i: min(b_i + batch_size, B)]]
                b_edges = old_data["edges"][idx[b_i: min(b_i + batch_size, B)]]
                b_actions = old_data["actions"][idx[b_i: min(b_i + batch_size, B)]]
                b_log_probs = old_data["log_probs"][idx[b_i: min(b_i + batch_size, B)]]
                b_advantages = advantages[idx[b_i: min(b_i + batch_size, B)]]
                b_returns = returns[idx[b_i: min(b_i + batch_size, B)]]


                h = gnn(b_nodes, b_edges)

                log_probs = 0
                entropies = []
                for i, actor in enumerate(actors):
                    mean, std = actor(h[:,i,:])
                    dist = Normal(mean, std)

                    log_probs += dist.log_prob(b_actions[:,i]).sum(dim=-1)
                    entropies.append(dist.entropy().sum(dim=-1))
                
                entropy = torch.stack(entropies, dim=1).sum(dim=-1).mean()
                values = critic(torch.flatten(h, start_dim=-2)).squeeze(-1)

                ratios = torch.exp(log_probs - b_log_probs)

                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * b_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values, b_returns)

                total_loss = actor_loss + (C1 * critic_loss) + (C2 * entropy)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=.5)
                optimizer.step()

        print(f"Epoch {epoch} complete. Reward: {reward}")

if __name__ == "__main__":
    train_active_orchestration(engine, gnn, actors, critic, optimizer, epochs=1, old_data_size=10)