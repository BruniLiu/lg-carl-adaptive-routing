from lgcarl.config import load_config
from lgcarl.env.routing_env import RoutingEnv
from lgcarl.graph.topology import build_topology
from lgcarl.rl.agent import DQNAgent


def test_agent_selects_and_updates():
    config = load_config("configs/default.yaml")
    config["env"]["episode_length"] = 4
    config["model"]["hidden_dim"] = 8
    config["model"]["node_embedding_dim"] = 4
    config["dqn"]["batch_size"] = 2
    graph = build_topology(config)
    env = RoutingEnv(graph, config)
    agent = DQNAgent(env.num_nodes, config["model"], config["dqn"], device="cpu")

    obs = env.reset(seed=7)
    for _ in range(3):
        action = agent.select_action(obs, training=True)
        next_obs, reward, done, _ = env.step(action)
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
    assert agent.update() is not None

