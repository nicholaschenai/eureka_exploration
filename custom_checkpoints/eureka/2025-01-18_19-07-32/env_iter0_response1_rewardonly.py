@torch.jit.script
def compute_reward(root_states: torch.Tensor, targets: torch.Tensor, dt: float) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    torso_position = root_states[:, 0:3]
    velocity = root_states[:, 7:10]
    
    # Compute the direction to the target on the ground plane
    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    normalized_to_target = torch.nn.functional.normalize(to_target, dim=-1)
    
    # Speed component: project velocity onto the target direction
    speed_along_target = (velocity * normalized_to_target).sum(dim=-1)
    
    # Maintain upright position (reward for being close to upright)
    # Assuming z-axis (2nd index) is "up" in torso_position
    upright_bonus = torch.exp(-torch.abs(torso_position[:, 2] - 1.0))
    
    # Calculate overall reward
    speed_reward_weight = 1.0
    upright_bonus_weight = 0.5
    
    # Apply exponential transformation
    speed_temp = 0.1
    upright_temp = 0.1

    speed_reward = speed_reward_weight * torch.exp(speed_temp * speed_along_target)
    upright_reward = upright_bonus_weight * torch.exp(upright_temp * upright_bonus)

    total_reward = speed_reward + upright_reward

    rewards = {
        "speed_reward": speed_reward,
        "upright_reward": upright_reward
    }

    return total_reward, rewards
