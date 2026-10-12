"""LLM strategiya qatı ("məşqçi").

Sürətli qat (agents.py) hər tickdə qərar verir və HEÇ VAXT gözləmir.
Yavaş qat (bu fayl) hər bir neçə saniyədən bir matçın xülasəsini LLM-ə göndərir, LLM 3 rejimdən birini seçir:
    attack / balanced / defend
və seçilən rejim sürətli qatın parametrlərini dəyişir. LLM gec cavab versə və ya xəta versə, komanda əvvəlki rejimlə oynamağa davam edir.

İstifadə (real matç, AWS Bedrock). İki yol var:
    MantleCoach  - yeni Claude modelləri (məs. anthropic.claude-haiku-4-5), "anthropic[bedrock]" paketi
    BedrockCoach - Converse API (köhnə Claude, Amazon Nova və s.), boto3
    $env:BEDROCK_MODEL_ID="<model ID>"; $env:AWS_REGION="us-east-1"
    team = CoachedTeam(coach=MantleCoach(), mode="threaded")
Simulyatorda sınaq:
    python run_match.py --coach mock       (AWS lazım deyil)
    python run_match.py --coach ollama     (PULSUZ lokal LLM, Ollama lazımdır)
    python run_match.py --coach mantle     (yeni Claude, AWS hesabı lazımdır)
    python run_match.py --coach bedrock    (Converse API, AWS hesabı + boto3 lazımdır)
"""
import json
import os
import random
import re
import threading

from agents import RuleAgent, PARAMS

# Rejimlərin sürətli qata təsiri: bunlar agents.py-dəki PARAMS-ın üstünə yazılır
MODES = {
    "balanced": {},
    "attack":   dict(shoot_dist=45, support_ahead=30, def_line=0.8, def_zone=0.9),
    "defend":   dict(shoot_dist=32, support_ahead=12, def_line=0.3, def_zone=0.45),
}

SYSTEM_PROMPT = """You are the coach of a 5-a-side football team. Every few seconds you get a JSON summary and must pick ONE team mode.
Modes: "attack" (push forward, shoot early), "balanced" (default), "defend" (drop deep, protect the goal).
Fields: result ("winning"/"losing"/"tied" for OUR team), goal_diff (our goals minus theirs), phase ("early"/"middle"/"late"),
time_left_pct, ball_zone, possession ("us"/"not_us"), our_avg_x, their_avg_x, allowed_modes.
RULES: choose ONLY from allowed_modes. If result is "losing" prefer attack. If "winning" and phase is "late" prefer defend. Otherwise prefer balanced.
Examples:
{"result":"losing","goal_diff":-1,"phase":"late","allowed_modes":["attack","balanced"]} -> {"mode":"attack","reason":"losing, little time"}
{"result":"winning","goal_diff":1,"phase":"late","allowed_modes":["defend","balanced"]} -> {"mode":"defend","reason":"protect the lead"}
{"result":"tied","goal_diff":0,"phase":"early","allowed_modes":["balanced"]} -> {"mode":"balanced","reason":"early, even game"}
Reply with ONLY one JSON object: {"mode": "...", "reason": "<10 words"}"""


def allowed_modes(diff, left):
    """Məntiqli rejimlər: geridəyiksə müdafiəyə keçmək, öndə və vaxt azdırsa hücum etmək yolverilməzdir."""
    if diff < 0:
        return ["attack", "balanced"]
    if diff > 0:
        return ["defend", "balanced"] if left <= 40 else ["balanced", "attack"]
    return ["attack", "balanced"] if left <= 25 else ["balanced"]


def build_summary(us, them, left, ball_zone="middle", possession="us", our_avg_x=50, their_avg_x=50):
    diff = us - them
    return {
        "result": "winning" if diff > 0 else ("losing" if diff < 0 else "tied"),
        "goal_diff": diff,
        "phase": "early" if left > 66 else ("middle" if left > 33 else "late"),
        "time_left_pct": left,
        "score_us": us, "score_them": them,
        "ball_zone": ball_zone, "possession": possession,
        "our_avg_x": our_avg_x, "their_avg_x": their_avg_x,
        "allowed_modes": allowed_modes(diff, left),
    }


def summarize(s):
    """Oyunçunun gördüyü vəziyyətdən (hamını görür) LLM üçün kiçik xülasə çıxarır."""
    mx = s.get("max_ticks", 1800)
    own = [s["me"]["pos"]] + [m["pos"] for m in s["mates"]]
    opp = [o["pos"] for o in s["opps"]]
    bx = s["ball"][0]
    sc = s.get("score", (0, 0))
    return build_summary(
        sc[0], sc[1], max(0, round(100 * (1 - s["tick"] / mx))),
        "our_third" if bx < 33 else ("middle" if bx < 67 else "their_third"),
        "us" if s["team_has_ball"] else "not_us",
        round(sum(p[0] for p in own) / len(own)),
        round(sum(p[0] for p in opp) / len(opp)) if opp else 50)


def parse_decision(text):
    """LLM cavabından {"mode","reason"} çıxarır. Xarab cavabda None qaytarır (komanda əvvəlki rejimdə qalır)."""
    if not text:
        return None
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            mode = str(d.get("mode", "")).strip().lower()
            if mode in MODES:
                return {"mode": mode, "reason": str(d.get("reason", ""))[:80]}
        except Exception:
            pass
    low = text.lower()
    found = [k for k in MODES if k in low]
    return {"mode": found[0], "reason": "mətndən"} if len(found) == 1 else None


# ---------------- məşqçilər ----------------
class MockCoach:
    """AWS-siz sınaq üçün sadə qayda: hesaba və qalan vaxta baxır."""
    def decide(self, x):
        diff, left = x["score_us"] - x["score_them"], x["time_left_pct"]
        if diff < 0 and left < 60:
            return {"mode": "attack", "reason": "geridəyik"}
        if diff > 0 and left < 40:
            return {"mode": "defend", "reason": "öndəyik, vaxt azdır"}
        if diff == 0 and left < 20:
            return {"mode": "attack", "reason": "bərabərlik, son dəqiqələr"}
        return {"mode": "balanced", "reason": "normal"}


class RandomCoach:
    """Müqayisə üçün: təsadüfi rejim seçir (rejim dəyişməsinin özünün faydası olmadığını yoxlamaq üçün)."""
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def decide(self, x):
        return {"mode": self.rng.choice(list(MODES)), "reason": "təsadüfi"}


class BedrockCoach:
    """AWS Bedrock (Converse API) ilə LLM məşqçi. Model ID-ni Bedrock konsolundan kopyalayın."""
    def __init__(self, model_id=None, region=None, client=None, timeout=4.0):
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
        if not self.model_id:
            raise ValueError("BEDROCK_MODEL_ID təyin edilməyib (Bedrock konsolundan model/inference profile ID kopyalayın).")
        if client is None:
            import boto3
            from botocore.config import Config
            client = boto3.client(
                "bedrock-runtime",
                region_name=region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
                config=Config(read_timeout=timeout, connect_timeout=2, retries={"max_attempts": 0}),
            )
        self.client = client

    def decide(self, x):
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": SYSTEM_PROMPT}],
            messages=[{"role": "user", "content": [{"text": json.dumps(x)}]}],
            inferenceConfig={"maxTokens": 120, "temperature": 0.0},
        )
        return parse_decision(resp["output"]["message"]["content"][0]["text"])


class OllamaCoach:
    """PULSUZ lokal LLM (Ollama), öz kompüterinizdə işləyir, internet və AWS lazım deyil.
    Quraşdırma: https://ollama.com -> yüklə, sonra terminalda: ollama pull llama3.2:3b"""
    def __init__(self, model=None, url=None, timeout=15.0):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self._warmup()

    def _post(self, path, payload, timeout):
        import urllib.request
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def _warmup(self):
        """Modeli əvvəlcədən yaddaşa yüklə (ilk sorğu yavaş olmasın) və 30 dəq yaddaşda saxla."""
        try:
            self._post("/api/generate", {"model": self.model, "keep_alive": "30m"}, max(self.timeout, 180))
        except Exception:
            pass   # Ollama işləmirsə xəta decide() zamanı görünəcək

    def decide(self, x):
        schema = {"type": "object", "required": ["mode", "reason"],
                  "properties": {"mode": {"type": "string", "enum": list(MODES)}, "reason": {"type": "string"}}}
        payload = {"model": self.model, "stream": False, "format": schema, "keep_alive": "30m",
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": json.dumps(x)}],
                   "options": {"temperature": 0}}
        try:
            data = self._post("/api/chat", payload, self.timeout)
        except Exception as e:
            if "400" not in str(e):
                raise
            payload["format"] = "json"       # köhnə Ollama schema-nı dəstəkləmir
            data = self._post("/api/chat", payload, self.timeout)
        return parse_decision(data["message"]["content"])


class MantleCoach:
    """"Claude in Amazon Bedrock" (Messages API) ilə LLM məşqçi: yeni Claude modelləri üçün (məs. anthropic.claude-haiku-4-5).
    Quraşdırma: pip install -U "anthropic[bedrock]" """
    def __init__(self, model_id=None, region=None, client=None, timeout=4.0):
        self.model_id = model_id or os.environ.get("BEDROCK_MODEL_ID")
        if not self.model_id:
            raise ValueError("BEDROCK_MODEL_ID təyin edilməyib (məs. anthropic.claude-haiku-4-5).")
        if client is None:
            from anthropic import AnthropicBedrockMantle
            client = AnthropicBedrockMantle(
                aws_region=region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
                timeout=timeout, max_retries=0)
        self.client = client

    def decide(self, x):
        resp = self.client.messages.create(
            model=self.model_id, max_tokens=120, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(x)}])
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), "")
        return parse_decision(text)


# ---------------- komanda ----------------
class CoachedTeam:
    """RuleAgent + LLM məşqçi. decide() heç vaxt LLM-i gözləmir.
    mode="threaded":  LLM ayrı axında çağırılır (real matç).
    mode="simulated": çağırış sinxron gedir, cavab latency_ticks tick sonra tətbiq olunur (simulyator sınağı)."""
    def __init__(self, agent=None, coach=None, interval_ticks=100, latency_ticks=20, mode="simulated"):
        self.agent = agent or RuleAgent(PARAMS)
        self.base = dict(self.agent.p)
        self.coach = coach or MockCoach()
        self.interval, self.latency, self.mode = interval_ticks, latency_ticks, mode
        self.history, self.errors, self.guard_hits = [], 0, 0
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.current = "balanced"
        self.agent.p = dict(self.base)
        self.last_req, self.pending, self._busy, self._result, self._last_tick = -10**9, None, False, None, -1

    def decide(self, s):
        t = s["tick"]
        if t != self._last_tick:
            if t < self._last_tick:
                self._reset()
            self._last_tick = t
            self._update(s, t)
        return self.agent.decide(s)

    def _safe_call(self, summary):
        try:
            res = self.coach.decide(summary)
            if res:
                res = dict(res)
                res["_allowed"] = summary.get("allowed_modes")
            return res
        except Exception:
            self.errors += 1
            return None

    def _worker(self, summary):
        res = self._safe_call(summary)
        with self._lock:
            self._result, self._busy = res, False

    def _update(self, s, t):
        if self.mode == "threaded":
            with self._lock:
                res, self._result = self._result, None
            if res:
                self._apply(t, res)
        elif self.pending and t >= self.pending[0]:
            self._apply(t, self.pending[1])
            self.pending = None
        if t - self.last_req >= self.interval and not self._busy and self.pending is None:
            self.last_req = t
            summary = summarize(s)
            if self.mode == "threaded":
                self._busy = True
                threading.Thread(target=self._worker, args=(summary,), daemon=True).start()
            else:
                res = self._safe_call(summary)
                if res:
                    self.pending = (t + self.latency, res)

    def _apply(self, t, res):
        mode, reason = res.get("mode"), res.get("reason", "")
        if mode not in MODES:
            return
        allowed = res.get("_allowed")
        if allowed and mode not in allowed:          # mühafizə: məntiqsiz LLM qərarı rədd edilir
            self.guard_hits += 1
            new = "balanced" if "balanced" in allowed else allowed[0]
            reason, mode = f"[mühafizə] LLM {mode} dedi -> {new}", new
        self.current = mode
        self.agent.p = {**self.base, **MODES[mode]}
        self.history.append((t, mode, reason))


def make_coached_team(kind="mock", **kw):
    coach = {"mock": MockCoach, "random": RandomCoach, "bedrock": BedrockCoach, "mantle": MantleCoach, "ollama": OllamaCoach}[kind]()
    return CoachedTeam(coach=coach, **kw)
