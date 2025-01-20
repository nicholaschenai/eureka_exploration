@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract relevant state information
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute direction to target and normalize
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    normalized_to_target = torch.nn.functional.normalize(to_target, dim=-1)
    
    # Speed component: Project velocity onto the target direction
    speed_along_target = (velocity * normalized_to_target).sum(dim=-1)
    
    # Maintain upright position (reward for being close to upright)
    target_z = 1.0
    z_deviation = torch.abs(torso_position[:, 2] - target_z)
    upright_bonus = torch.maximum(torch.tensor([0.0], device=torso_position.device), 1.0 - z_deviation)
    
    # Stability penalty 
    stability_penalty_weight = 1.5
    stability_penalty = stability_penalty_weight * z_deviation
    
    # Calculate overall reward
    speed_reward_weight = 1.3
    upright_bonus_weight = 1.5
    
    # Temperature parameters for transformations
    speed_temp = 0.15
    upright_temp = 1.0
    
    speed_reward = speed_reward_weight * torch.exp(speed_temp * speed_along_target)
    upright_reward = upright_bonus_weight * torch.exp(upright_temp * upright_bonus)
    
    # Total reward
    total_reward = speed_reward + upright_reward - stability_penalty
    
    rewards = {
        "speed_reward": speed_reward,
        "upright_reward": upright_reward,
        "stability_penalty": stability_penalty
    }

    return total_reward, rewards
