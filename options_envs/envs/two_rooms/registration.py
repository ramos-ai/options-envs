from gymnasium.envs.registration import register, registry


def register_two_rooms_envs():
    env_id = "OptionsEnv/TwoRooms-v0"
    if env_id not in registry:
        register(id=env_id, entry_point="options_envs.envs.two_rooms.env:TwoRoomsEnv")
