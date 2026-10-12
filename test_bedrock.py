"""LLM bağlantısını və qərar keyfiyyətini oyun olmadan yoxla (12 sınaq vəziyyəti).
  python test_bedrock.py ollama     (PULSUZ lokal LLM)
  python test_bedrock.py mantle     (yeni Claude, AWS)
  python test_bedrock.py bedrock    (Converse API, AWS)
  python test_bedrock.py mock       (qaydalı məşqçi, istinad)
Hər vəziyyətdə LLM-in seçdiyi rejim "məntiqli rejimlər" siyahısında olmalıdır. Xal: neçəsi düzgün.
Əvvəl mühit dəyişənləri (mantle/bedrock üçün): BEDROCK_MODEL_ID, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
"""
import os
import sys
import time

from llm_coach import BedrockCoach, MantleCoach, OllamaCoach, MockCoach, build_summary

kind = sys.argv[1] if len(sys.argv) > 1 else "ollama"
if kind == "ollama":
    print(f"Yol: ollama | model: {os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')}\nModel yaddaşa yüklənir (ilk dəfə 1-2 dəqiqə çəkə bilər, gözləyin)...")
elif kind != "mock":
    print(f"Yol: {kind} | model: {os.environ.get('BEDROCK_MODEL_ID')} | region: {os.environ.get('AWS_REGION')}")

# (bizim qol, rəqib qol, qalan vaxt %, izah)
CASES = [(0, 0, 95, "başlanğıc"), (0, 0, 60, "bərabər, orta"), (0, 0, 15, "bərabər, son dəqiqələr"),
         (0, 1, 70, "geridə, erkən"), (0, 1, 20, "geridə, son dəqiqələr"), (1, 3, 40, "çox geridə"),
         (1, 0, 80, "öndə, erkən"), (1, 0, 30, "öndə, son dəqiqələr"), (2, 0, 10, "öndə, qalib vəziyyət"),
         (2, 2, 30, "bərabər, gec"), (3, 1, 50, "öndə, orta"), (0, 2, 10, "geridə, vaxt bitir")]
try:
    coach = {"mantle": MantleCoach, "bedrock": BedrockCoach, "ollama": OllamaCoach, "mock": MockCoach}[kind]
    coach = coach() if kind == "mock" else coach(timeout=60 if kind == "ollama" else 15)
    ok, total_t = 0, 0.0
    for us, them, left, note in CASES:
        x = build_summary(us, them, left)
        t = time.time()
        d = coach.decide(x)
        dt = time.time() - t
        total_t += dt
        good = bool(d) and d["mode"] in x["allowed_modes"]
        ok += good
        print(f"  {'OK ' if good else 'XƏTA'} {us}:{them}, qalan {left:3d}% ({note:24s}) -> {d['mode'] if d else None:9s} (məntiqli: {'/'.join(x['allowed_modes'])})  {dt:.1f} san")
    print(f"\nNƏTİCƏ: {ok}/{len(CASES)} məntiqli qərar | orta gecikmə {total_t / len(CASES):.1f} san")
    print("Qeyd: oyunda səhv qərarlar 'mühafizə' tərəfindən rədd edilir, amma LLM nə qədər düz seçirsə, o qədər faydalıdır.")
except Exception as e:
    msg = f"{type(e).__name__}: {e}"
    print("\nXƏTA:", msg[:600])
    low = msg.lower()
    if "refused" in low or "urlopen" in low or "10061" in low:
        print("-> Ollama işləmir: Ollama proqramını açın (və ya terminalda: ollama serve), modeli yükləyin: ollama pull llama3.2:3b")
    elif "credential" in low or "unauthorized" in low or "security token" in low or "signature" in low:
        print("-> AWS giriş açarları (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY) səhv və ya yoxdur.")
    elif "accessdenied" in low or "not authorized" in low or "access denied" in low or "403" in low:
        print("-> İcazə yoxdur: IAM istifadəçisinə Bedrock icazəsi verin, Bedrock konsolunda Anthropic modelinin 'use case' formunu doldurun, regionun modeli dəstəklədiyini yoxlayın.")
    elif "region" in low or "endpoint" in low:
        print("-> AWS_REGION təyin edin (məs. us-east-1) və modelin həmin regionda olduğunu yoxlayın.")
    elif "model" in low or "validation" in low or "404" in low:
        print("-> Model ID səhvdir və ya bu yolla dəstəklənmir: digər yolu yoxlayın (mantle <-> bedrock).")
    elif "timeout" in low or "timed out" in low:
        print("-> Vaxt bitdi: internet və ya model yavaşdır.")
    sys.exit(1)
