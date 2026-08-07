# Adding a New Environment

When adding a new environment to `options-envs`, keep the implementation isolated under `options_envs/envs/<env_name>/` and register it through the central package registration flow.

Minimum expectations:

- Add the environment package under `options_envs/envs/<env_name>/`.
- Add tasks and assets inside the environment directory when needed.
- Register the Gymnasium ID in the environment-specific `registration.py`.
- Add tests covering registration and the basic Gymnasium contract.
- Add environment documentation in `docs/envs/<env_name>.md`.

Example:

- `docs/envs/four_rooms.md`

Use [`docs/envs/template.md`](envs/template.md) as the starting point for the
environment page.
