from options_envs.envs.pinball.registration import register_pinball_envs
from options_envs.envs.four_rooms.registration import register_four_rooms_envs
from options_envs.envs.two_rooms.registration import register_two_rooms_envs


def register_envs():
    register_pinball_envs()
    register_four_rooms_envs()
    register_two_rooms_envs()
