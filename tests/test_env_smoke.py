from lgcarl.config import load_config
from lgcarl.env.routing_env import RoutingEnv
from lgcarl.graph.topology import build_topology


def test_env_reset_and_step():
    config = load_config("configs/default.yaml")
    config["env"]["episode_length"] = 3
    graph = build_topology(config)
    env = RoutingEnv(graph, config)
    obs = env.reset(seed=123)
    assert obs["link_features"].shape[1] == 7
    assert len(obs["candidate_paths"]) == config["env"]["k_paths"]
    next_obs, reward, done, info = env.step(0)
    assert isinstance(reward, float)
    assert "delay" in info
    assert next_obs["path_features"].shape[0] == config["env"]["k_paths"]

