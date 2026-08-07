from gymnasium.envs.registration import register, registry


def register_four_rooms_envs():
    env_id = "OptionsEnv/FourRooms-v0"
    if env_id not in registry:
        register(id=env_id, entry_point="options_envs.envs.four_rooms.env:FourRoomsEnv")
