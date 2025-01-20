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
    # Assume target_z indicates optimal standing height
    target_z = 1.0
    z_deviation = torch.abs(torso_position[:, 2] - target_z)
    upright_bonus = torch.exp(-z_deviation)  # More sensitive upright component
    
    # Penalize deviations from stable orientations (e.g., falling)
    # Assuming stable upright when z_deviation is small
    stability_penalty_weight = 1.0
    stability_penalty = stability_penalty_weight * z_deviation
    
    # Calculate overall reward
    speed_reward_weight = 1.0
    upright_bonus_weight = 0.5
    stability_penalty_weight = 0.2
    
    # Temperature parameters for transformations
    speed_temp = 0.1
    upright_temp = 0.2
    
    speed_reward = speed_reward_weight * torch.exp(speed_temp * speed_along_target)
    upright_reward = upright_bonus_weight * torch.exp(upright_temp * upright_bonus)
    
    # Subtract stabilization penalty
    total_reward = speed_reward + upright_reward - stability_penalty_weight * stability_penalty

    rewards = {
        "speed_reward": speed_reward,
        "upright_reward": upright_reward,
        "stability_penalty": stability_penalty
    }

    return total_reward, rewards
