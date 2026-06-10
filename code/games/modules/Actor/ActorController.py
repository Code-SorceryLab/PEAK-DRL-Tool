from __future__ import annotations


class ActorController:
    """Ties a MotorState + brain + ordered abilities together.

    update(dt) runs ONE decision+ability pass (sets velocity only). The host
    (MeatboyCore via ModularPhysicsManager) then applies gravity, integrates
    position, resolves collisions, and writes contact flags back to the state.
    """
    def __init__(self, state, brain, abilities, ctx):
        self.state = state
        self.brain = brain
        self.abilities = list(abilities)
        self.ctx = ctx

    def update(self, dt: float) -> None:
        s = self.state
        s.gravity_scale = 1.0          # reset BEFORE abilities set it this frame
        # 1. brain fills intents
        self.brain.decide(s)
        # 2. abilities run in registered order (each mutates MotorState)
        for ab in self.abilities:
            if ab.active:
                ab.update(s, self.ctx, dt)
        # 3. decay motor-level lockout (read by Move to preserve wall-jump push)
        if s.air_lockout > 0:
            s.air_lockout -= 1

    # --- observation: ability fields appended in registered order ---
    def obs_field_names(self):
        names = []
        for ab in self.abilities:
            names.append(f"{ab.name}_active")
            names.extend(ab.obs_spec())
        return names

    def write_obs(self):
        vals = []
        for ab in self.abilities:
            vals.append(1.0 if ab.active else 0.0)
            vals.extend(ab.write_obs(self.state))
        return vals
