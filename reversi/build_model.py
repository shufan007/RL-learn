
import torch
from model_deploy.policies import CustomCNN, ActorCriticPolicy

def build_model(state_dict_path, features_extractor_kwargs, observation_space=(4,8,8)):

    # features_extractor_kwargs=dict(features_dim=1024,
    #                                net_arch=[64, 128, 256],
    #                                # net_arch=[64, 128, 128],
    #                                # net_arch=[32, 64, 128],
    #                                kernel_size=3,
    #                                stride=1,
    #                                padding='same',
    #                                is_batch_norm=False),

    action_space = [observation_space[1]*observation_space[2]]

    policy_model = ActorCriticPolicy(
                    observation_space = observation_space,
                    action_space = action_space,
                    net_arch=[256, 512],
                    features_extractor_class = CustomCNN,
                    features_extractor_kwargs = features_extractor_kwargs,
                    normalize_images= True,
    )

    return policy_model

