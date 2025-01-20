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
    
    # Maintain upright position (reward should vary with deviation from 1.0 height)
    upright_penalty = torch.abs(torso_position[:, 2] - 1.0)
    
    # Updated upright reward to be more sensitive
    upright_reward = 1.0 - torch.clamp(upright_penalty, 0.0, 1.0)
    
    # Calculate overall reward
    speed_reward_weight = 1.0
    upright_reward_weight = 0.6
    
    # Apply exponential transformation with adjusted temperatures
    speed_temp = 0.1  # slightly increase to explore more
    upright_temp = 0.5  # increase to emphasize this component

    speed_reward = speed_reward_weight * torch.exp(speed_temp * speed_along_target)
    upright_transformed = torch.exp(upright_temp * upright_reward)

    total_reward = speed_reward + upright_transformed

    rewards = {
        "speed_reward": speed_reward,
        "upright_reward": upright_transformed
    }

    return total_reward, rewards
