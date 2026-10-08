import os
import time
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("multistream.rerank")


class VLMReranker:
    """
    Vision-Language Model (VLM) Candidate Reranker.
    Supports Gemini and Claude cloud vision APIs to judge top candidates,
    or runs in local 'off' mode for 100% offline edge privacy.
    """
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or os.environ.get("RERANK_PROVIDER", "off")).lower()
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")

        if self.provider == "claude" and not self.anthropic_key:
            logger.warning("[RERANK] ANTHROPIC_API_KEY missing; fallback to RERANK_PROVIDER=off")
            self.provider = "off"
        elif self.provider == "gemini" and not self.gemini_key:
            logger.warning("[RERANK] GEMINI_API_KEY missing; fallback to RERANK_PROVIDER=off")
            self.provider = "off"

    def judge_image(self, query: str, image_path: Path | str) -> Tuple[bool, float, str]:
        """
        Judge whether a snapshot contains the object described in query.
        Returns: (is_relevant: bool, confidence: float, explanation: str)
        """
        img_p = Path(image_path)
        if not img_p.exists():
            return False, 0.0, f"Image not found: {img_p}"

        if self.provider == "off":
            return True, 0.5, "Rerank off (local bypass)"

        t0 = time.perf_counter()

        # Cloud egress warning
        logger.info(f"[PRIVACY NOTICE] Transmitting snapshot {img_p.name} to cloud VLM provider '{self.provider}'.")

        try:
            if self.provider == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))
                
                with open(img_p, "rb") as f:
                    img_data = f.read()

                prompt = (
                    f"Look at this surveillance crop/snapshot. Does it clearly depict '{query}'? "
                    "Respond with exactly one line in format: YES/NO | CONFIDENCE (0.0 to 1.0) | BRIEF REASON"
                )
                response = model.generate_content([
                    {"mime_type": "image/jpeg", "data": img_data},
                    prompt
                ])
                text = response.text.strip()
                parts = text.split("|")
                is_match = "yes" in parts[0].strip().lower()
                conf = float(parts[1].strip()) if len(parts) > 1 else (0.9 if is_match else 0.1)
                reason = parts[2].strip() if len(parts) > 2 else text
                return is_match, conf, reason

            elif self.provider == "claude":
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_key)
                with open(img_p, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")

                prompt = (
                    f"Look at this surveillance crop/snapshot. Does it clearly depict '{query}'? "
                    "Respond with exactly one line in format: YES/NO | CONFIDENCE (0.0 to 1.0) | BRIEF REASON"
                )
                msg = client.messages.create(
                    model=os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
                    max_tokens=60,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_data}},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                text = msg.content[0].text.strip()
                parts = text.split("|")
                is_match = "yes" in parts[0].strip().lower()
                conf = float(parts[1].strip()) if len(parts) > 1 else (0.9 if is_match else 0.1)
                reason = parts[2].strip() if len(parts) > 2 else text
                return is_match, conf, reason

        except Exception as e:
            logger.error(f"[RERANK] VLM call failed: {e}")
            return False, 0.0, f"Error: {e}"

        return True, 0.5, "Unknown provider"

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: int = 10
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Rerank top candidates using VLM judgments.
        """
        if self.provider == "off" or not candidates:
            return candidates, {"provider": "off", "reranked_count": 0, "latency_ms": 0.0}

        t0 = time.perf_counter()
        to_rerank = candidates[:top_k]
        remaining = candidates[top_k:]

        judged = []
        for cand in to_rerank:
            snap_path = getattr(cand, "snapshot_path", None)
            if snap_path and Path(snap_path).exists():
                is_match, conf, reason = self.judge_image(query, snap_path)
                # Boost/weight score by VLM confidence
                new_score = cand.score * 0.4 + (conf if is_match else 0.0) * 0.6
                cand.score = round(new_score, 4)
            judged.append(cand)

        judged.sort(key=lambda x: x.score, reverse=True)
        final_results = judged + remaining
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return final_results, {
            "provider": self.provider,
            "reranked_count": len(to_rerank),
            "latency_ms": round(elapsed_ms, 2)
        }
