"""
In a production deployment these actions would call iptables/nftables APIs or
Suricata fast-pattern hooks or layer7 filter etc. Here i will try to provide a clean simulation layer with
full auditability as much as possible
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms",
]
IDX2ATTACK = {i: a for i, a in enumerate(ATTACK_CATEGORIES)}

class IPSAction(Enum):
    ALLOW      = auto()
    ALERT      = auto()
    RATE_LIMIT = auto()
    BLOCK      = auto()

SEVERITY_MAP: Dict[str, IPSAction] = {
    "Normal":        IPSAction.ALLOW,
    "Reconnaissance": IPSAction.ALERT,
    "Analysis":      IPSAction.ALERT,
    "Fuzzers":       IPSAction.RATE_LIMIT,
    "Generic":       IPSAction.RATE_LIMIT,
    "DoS":           IPSAction.BLOCK,
    "Exploits":      IPSAction.BLOCK,
    "Backdoor":      IPSAction.BLOCK,
    "Shellcode":     IPSAction.BLOCK,
    "Worms":         IPSAction.BLOCK,
}

@dataclass
class IPSEvent:
    timestamp:       float
    predicted_class: str
    confidence:      float
    action:          IPSAction
    raw_probs:       List[float] = field(default_factory=list)
    source_id:       Optional[str] = None

class IPSEngine:


    def __init__(
        self,
        model: nn.Module,
        confidence_threshold: float = 0.85,
        rate_limit_threshold: float = 0.70,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.conf_thresh  = confidence_threshold
        self.rate_thresh  = rate_limit_threshold
        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.eval().to(self.device)
        self._event_log: List[IPSEvent] = []

    def _decide_action(
        self,
        attack_name: str,
        confidence: float,
    ) -> IPSAction:
        
        if attack_name == "Normal":
            return IPSAction.ALLOW

        severity_action = SEVERITY_MAP.get(attack_name, IPSAction.ALERT)

        if confidence < self.rate_thresh:
            return IPSAction.ALERT
        elif confidence < self.conf_thresh:
            if severity_action == IPSAction.BLOCK:
                return IPSAction.RATE_LIMIT
            return severity_action
        else:
            return severity_action

    @torch.no_grad()
    def process_batch(
        self,
        X: torch.Tensor,
        source_ids: Optional[List[str]] = None,
    ) -> List[IPSEvent]:

        X = X.to(self.device)
        logits = self.model(X)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
        confs  = probs[np.arange(len(preds)), preds]

        events = []
        for i in range(len(preds)):
            attack_name = IDX2ATTACK[preds[i]]
            confidence  = float(confs[i])
            action      = self._decide_action(attack_name, confidence)
            src         = source_ids[i] if source_ids else None

            event = IPSEvent(
                timestamp       = time.time(),
                predicted_class = attack_name,
                confidence      = confidence,
                action          = action,
                raw_probs       = probs[i].tolist(),
                source_id       = src,
            )
            events.append(event)
            self._event_log.append(event)

            if action != IPSAction.ALLOW:
                logger.info(
                    "[IPS] src=%-15s class=%-15s conf=%.3f action=%s",
                    src or "unknown", attack_name, confidence, action.name,
                )

        return events

    def simulate_firewall_rule(self, event: IPSEvent) -> str:
       
        if event.action == IPSAction.BLOCK:
            src = event.source_id or "UNKNOWN_SRC"
            return (
                f"iptables -A INPUT -s {src} -j DROP "
                f"# Blocked {event.predicted_class} (conf={event.confidence:.3f})"
            )
        elif event.action == IPSAction.RATE_LIMIT:
            src = event.source_id or "UNKNOWN_SRC"
            return (
                f"iptables -A INPUT -s {src} -m limit --limit 10/min -j ACCEPT "
                f"# Rate-limited {event.predicted_class} (conf={event.confidence:.3f})"
            )
        return f"# No firewall rule needed: action={event.action.name}"

    def get_summary(self) -> Dict:
        
        if not self._event_log:
            return {}

        action_counts = {a.name: 0 for a in IPSAction}
        class_counts  = {c: 0 for c in ATTACK_CATEGORIES}
        for ev in self._event_log:
            action_counts[ev.action.name]    += 1
            class_counts[ev.predicted_class] += 1

        total = len(self._event_log)
        return {
            "total_events":    total,
            "action_breakdown": {k: v / total for k, v in action_counts.items()},
            "class_breakdown":  class_counts,
            "block_rate":       action_counts["BLOCK"] / total,
            "alert_rate":       action_counts["ALERT"] / total,
        }

    def print_summary(self):
        s = self.get_summary()
        if not s:
            print("No events processed yet.")
            return
        print("\n" + "" * 50)
        print("  IPS Engine Summary")
        print("" * 50)
        print(f"  Total events : {s['total_events']:,}")
        print("\n  Action breakdown:")
        for action, frac in s["action_breakdown"].items():
            bar = "█" * int(frac * 30)
            print(f"    {action:<12s}: {bar:<30s} {frac*100:5.1f}%")
        print("\n  Detected attack classes:")
        for cls, cnt in sorted(s["class_breakdown"].items(),
                                key=lambda x: -x[1]):
            if cnt > 0:
                print(f"    {cls:<20s}: {cnt:,}")
        print("" * 50)
