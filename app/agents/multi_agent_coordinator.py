"""
Advanced Multi-Agent Coordination Tracker
=========================================
Production-grade coordination engine with Theory-of-Mind belief tracking,
handoff chain analysis, message usefulness scoring, and dynamic coordination
score (0.0-1.0) across 5 weighted factors.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List
import math
import time


@dataclass
class BeliefState:
    """What one agent believes another agent knows."""
    knowledge_items: List[str] = field(default_factory=list)
    confidence: float = 1.0
    last_updated: float = 0.0
    update_count: int = 0

    def add_knowledge(self, item: str) -> None:
        if item not in self.knowledge_items:
            self.knowledge_items.append(item)
        self.confidence = 1.0
        self.last_updated = time.time()
        self.update_count += 1

    def decay(self, half_life_seconds: float = 60.0) -> None:
        if self.last_updated <= 0:
            return
        elapsed = time.time() - self.last_updated
        self.confidence = max(0.0, math.exp(-0.693 * elapsed / half_life_seconds))


class CoordinationTracker:
    """
    Tracks inter-agent communication quality, handoff success rates,
    theory-of-mind beliefs, and produces a dynamic coordination score.
    """

    _W_USEFULNESS = 0.30
    _W_HANDOFF_SUCCESS = 0.25
    _W_BELIEF_BREADTH = 0.15
    _W_COMM_BALANCE = 0.15
    _W_INFO_FLOW = 0.15

    def __init__(self):
        self.communication_log: List[Dict] = []
        self.handoffs: List[Dict] = []
        self.successful_handoffs: int = 0
        self.useful_messages: int = 0
        self.total_messages: int = 0
        self.agent_beliefs: Dict[str, Dict[str, BeliefState]] = defaultdict(dict)
        self._sent_counts: Dict[str, int] = defaultdict(int)
        self._recv_counts: Dict[str, int] = defaultdict(int)
        self._handoff_chains: List[List[Dict]] = []
        self._current_chain: List[Dict] = []

    def log_communication(self, from_agent: str, to_agent: str,
                          message: Dict, led_to_action: bool = False) -> None:
        """Log inter-agent communication with usefulness tracking."""
        self.total_messages += 1
        content = message.get("content", "")
        content_quality = min(1.0, len(str(content)) / 200.0)
        usefulness_score = (0.6 if led_to_action else 0.0) + 0.4 * content_quality

        entry = {
            "from": from_agent, "to": to_agent, "message": message,
            "led_to_action": led_to_action, "timestamp": time.time(),
            "usefulness_score": round(usefulness_score, 4),
        }
        self.communication_log.append(entry)

        if led_to_action:
            self.useful_messages += 1

        self._sent_counts[from_agent] += 1
        self._recv_counts[to_agent] += 1

        if to_agent not in self.agent_beliefs[from_agent]:
            self.agent_beliefs[from_agent][to_agent] = BeliefState()
        self.agent_beliefs[from_agent][to_agent].add_knowledge(str(content)[:256])

    def log_handoff(self, from_agent: str, to_agent: str, target: str,
                    success: bool, latency_ms: float = 0.0,
                    context_preserved: bool = True) -> None:
        """Track a handoff between agents with quality metadata."""
        record = {
            "from": from_agent, "to": to_agent, "target": target,
            "success": success, "timestamp": time.time(),
            "latency_ms": latency_ms, "context_preserved": context_preserved,
        }
        self.handoffs.append(record)
        if success:
            self.successful_handoffs += 1

        if self._current_chain and self._current_chain[-1]["to"] == from_agent:
            self._current_chain.append(record)
        else:
            if self._current_chain:
                self._handoff_chains.append(list(self._current_chain))
            self._current_chain = [record]

    def calculate_coordination_score(self) -> float:
        """
        Dynamic coordination score (0.0-1.0) using 5 weighted factors:
        1. Message usefulness rate (30%) - recency-weighted
        2. Handoff success rate (25%) - with context preservation bonus
        3. Theory-of-Mind belief breadth (15%)
        4. Communication balance (15%) - Gini-like
        5. Information flow diversity (15%)
        """
        if self.total_messages == 0 and not self.handoffs:
            return 0.0

        # Factor 1: Usefulness (recency-weighted)
        if self.total_messages > 0:
            recent = self.communication_log[-50:]
            w_useful = sum(
                e.get("usefulness_score", 0) * (0.9 ** (len(recent) - 1 - i))
                for i, e in enumerate(recent)
            )
            w_total = sum(0.9 ** (len(recent) - 1 - i) for i in range(len(recent)))
            usefulness = w_useful / max(1e-9, w_total)
        else:
            usefulness = 0.0

        # Factor 2: Handoff success rate + context preservation
        if self.handoffs:
            base_rate = self.successful_handoffs / len(self.handoffs)
            ctx = sum(1 for h in self.handoffs if h.get("context_preserved", True))
            ctx_rate = ctx / len(self.handoffs)
            handoff_success = 0.8 * base_rate + 0.2 * ctx_rate
        else:
            handoff_success = 0.0

        # Factor 3: Theory-of-Mind belief breadth
        active_pairs = sum(
            1 for targets in self.agent_beliefs.values()
            for bs in targets.values()
            if bs.confidence > 0.2 and bs.update_count > 0
        )
        belief_breadth = min(1.0, active_pairs / 6.0)

        # Factor 4: Communication balance
        all_agents = set(self._sent_counts) | set(self._recv_counts)
        if len(all_agents) >= 2:
            total = sum(self._sent_counts.values()) + sum(self._recv_counts.values())
            per_agent = [
                self._sent_counts.get(a, 0) + self._recv_counts.get(a, 0)
                for a in all_agents
            ]
            mean_msgs = total / len(all_agents)
            if mean_msgs > 0:
                mad = sum(abs(v - mean_msgs) for v in per_agent) / len(all_agents)
                comm_balance = max(0.0, 1.0 - mad / mean_msgs)
            else:
                comm_balance = 0.0
        else:
            comm_balance = 0.5 if self.total_messages > 0 else 0.0

        # Factor 5: Information flow diversity
        unique_edges = set()
        for entry in self.communication_log:
            unique_edges.add((entry["from"], entry["to"]))
        for h in self.handoffs:
            unique_edges.add((h["from"], h["to"]))
        info_flow = min(1.0, len(unique_edges) / 8.0)

        score = (
            usefulness * self._W_USEFULNESS
            + handoff_success * self._W_HANDOFF_SUCCESS
            + belief_breadth * self._W_BELIEF_BREADTH
            + comm_balance * self._W_COMM_BALANCE
            + info_flow * self._W_INFO_FLOW
        )
        return round(min(1.0, max(0.0, score)), 4)

    def get_coordination_report(self) -> Dict:
        """Full coordination report for dashboard and debugging."""
        score = self.calculate_coordination_score()
        active_pairs = sum(
            1 for targets in self.agent_beliefs.values()
            for bs in targets.values()
            if bs.confidence > 0.2 and bs.update_count > 0
        )
        unique_edges = set()
        for entry in self.communication_log:
            unique_edges.add((entry["from"], entry["to"]))

        all_chains = list(self._handoff_chains)
        if self._current_chain:
            all_chains.append(list(self._current_chain))
        chain_lengths = [len(c) for c in all_chains] if all_chains else [0]

        return {
            "coordination_score": score,
            "total_messages": self.total_messages,
            "useful_messages": self.useful_messages,
            "usefulness_rate": round(
                self.useful_messages / max(1, self.total_messages), 4),
            "total_handoffs": len(self.handoffs),
            "successful_handoffs": self.successful_handoffs,
            "handoff_success_rate": round(
                self.successful_handoffs / max(1, len(self.handoffs)), 4),
            "agents_communicating": sorted(set(self._sent_counts)),
            "theory_of_mind": {
                "active_belief_pairs": active_pairs,
                "total_knowledge_items": sum(
                    len(bs.knowledge_items)
                    for targets in self.agent_beliefs.values()
                    for bs in targets.values()
                ),
                "avg_belief_confidence": round(
                    sum(bs.confidence for targets in self.agent_beliefs.values()
                        for bs in targets.values()) / max(1, active_pairs), 4),
            },
            "handoff_chains": {
                "total_chains": len(all_chains),
                "max_chain_length": max(chain_lengths),
                "avg_chain_length": round(
                    sum(chain_lengths) / max(1, len(chain_lengths)), 2),
            },
            "unique_communication_edges": len(unique_edges),
        }

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self.communication_log.clear()
        self.handoffs.clear()
        self.successful_handoffs = 0
        self.useful_messages = 0
        self.total_messages = 0
        self.agent_beliefs.clear()
        self._sent_counts.clear()
        self._recv_counts.clear()
        self._handoff_chains.clear()
        self._current_chain.clear()
