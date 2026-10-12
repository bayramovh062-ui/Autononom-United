"""Təlim keçmiş RL modelini oyunda işlədən agent: RL oyunçu (default hücumçu #3) + qalanı qaydalı."""
from agents import RuleAgent, PARAMS
from rl_obs import obs_from_state, skill_to_cmd

FRAME_SKIP = 3   # təlimdəki kimi: hər 3 tickdə bir qərar


class RLTeam:
    def __init__(self, model, idx=3, base=None):
        if isinstance(model, str):
            from stable_baselines3 import PPO
            model = PPO.load(model, device="cpu")
        self.model, self.idx = model, idx
        self.base = base or RuleAgent(PARAMS)
        self.skill, self.last_tick = None, -99

    def decide(self, s):
        if s["me"]["index"] != self.idx:
            return self.base.decide(s)
        t = s["tick"]
        if self.skill is None or t < self.last_tick or t - self.last_tick >= FRAME_SKIP:
            a, _ = self.model.predict(obs_from_state(s), deterministic=True)
            self.skill, self.last_tick = int(a), t
        return skill_to_cmd(self.skill, s, self.base)
