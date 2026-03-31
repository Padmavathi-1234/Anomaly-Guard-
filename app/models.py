"""
AnomalyGuard — Pydantic v2 Models
Complete type definitions for the RL environment.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ───────────────────────────────────────────────────────────

class ThreatLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class HostStatus(str, Enum):
    ONLINE = "online"
    COMPROMISED = "compromised"
    ISOLATED = "isolated"
    RESTORED = "restored"
    OFFLINE = "offline"


class IncidentPhase(str, Enum):
    DETECTION = "detection"
    CONTAINMENT = "containment"
    ERADICATION = "eradication"
    RECOVERY = "recovery"
    COMPLETED = "completed"


class ActionType(str, Enum):
    TRIAGE_ALERT = "triage_alert"
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    DISABLE_ACCOUNT = "disable_account"
    PATCH_VULNERABILITY = "patch_vulnerability"
    REMOVE_PERSISTENCE = "remove_persistence"
    ROTATE_CREDENTIALS = "rotate_credentials"
    RESTORE_HOST = "restore_host"
    COLLECT_FORENSICS = "collect_forensics"
    ESCALATE_INCIDENT = "escalate_incident"
    QUERY_HOST = "query_host"          # NEW - for partial observability
    MONITOR = "monitor"                 # NEW - for baseline agents


# ── Data Models ─────────────────────────────────────────────────────

class MitreMapping(BaseModel):
    model_config = {"use_enum_values": True}

    technique_id: str = Field(..., description="MITRE ATT&CK technique ID")
    technique_name: str = Field(..., description="Human-readable technique name")
    tactic: str = Field(..., description="MITRE tactic category")


class SIEMAlert(BaseModel):
    model_config = {"use_enum_values": True}

    alert_id: str
    timestamp: str
    severity: str
    alert_type: str
    description: str
    source_host: str
    source_ip: str
    destination_ip: Optional[str] = None
    mitre_technique: Optional[MitreMapping] = None
    ioc_matches: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    is_true_positive: Optional[bool] = None  # HIDDEN from agent
    agent_classification: Optional[str] = None


class NetworkHost(BaseModel):
    model_config = {"use_enum_values": True}

    host_id: str
    hostname: str
    ip_address: str
    role: str  # workstation, server, database, domain_controller
    criticality: str  # low, medium, high, critical
    status: str = "online"
    c2_active: bool = False
    persistence: List[str] = Field(default_factory=list)
    vulnerabilities: List[str] = Field(default_factory=list)
    accounts: List[str] = Field(default_factory=list)
    business_impact: float = Field(ge=0.0, le=1.0, default=0.5)
    is_queried: bool = False           # NEW - for partial observability
    services: List[str] = Field(default_factory=list)  # NEW


class NetworkEvent(BaseModel):
    timestamp: str
    event_type: str
    source: str
    destination: Optional[str] = None
    details: str = ""


class ThreatIntel(BaseModel):
    attack_campaign: str = "Unknown"
    malicious_ips: List[str] = Field(default_factory=list)
    malicious_domains: List[str] = Field(default_factory=list)
    malicious_hashes: List[str] = Field(default_factory=list)
    known_cves: List[str] = Field(default_factory=list)
    threat_actor: str = "Unknown"


class ScoreBreakdown(BaseModel):
    action_correctness: float = 0.0
    reasoning_clarity: float = 0.0
    evidence_validity: float = 0.0
    risk_accuracy: float = 0.0
    overall: float = 0.0


# ── Observation ─────────────────────────────────────────────────────

class Observation(BaseModel):
    model_config = {"use_enum_values": True}

    task_id: int
    step: int
    max_steps: int
    alerts: List[SIEMAlert] = Field(default_factory=list)
    hosts: List[NetworkHost] = Field(default_factory=list)
    network_events: List[NetworkEvent] = Field(default_factory=list)
    threat_intel: Optional[ThreatIntel] = None
    incident_phase: str = "detection"
    score_so_far: float = 0.0
    time_remaining: int = 0
    difficulty: float = 0.5
    message: str = ""
    available_actions: List[str] = Field(default_factory=list)
    score_breakdown: Optional[ScoreBreakdown] = None


# ── Action Justification ───────────────────────────────────────────

class EvidenceItem(BaseModel):
    source: str = Field(..., description="Alert ID, host ID, or data source")
    content: str = Field(..., description="Specific data point observed")
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.7)


class RiskAssessment(BaseModel):
    threat_level: str = "unknown"
    confidence: float = 0.5
    potential_impact: str = "unknown"
    business_disruption_estimate: str = "unknown"


class AlternativeConsidered(BaseModel):
    action: str
    rejected_because: str = Field(..., min_length=10)


class ActionJustification(BaseModel):
    reasoning: str = ""
    evidence: List[EvidenceItem] = Field(default_factory=list)
    risk_assessment: Optional[RiskAssessment] = None
    alternatives_considered: List[AlternativeConsidered] = Field(
        default_factory=list
    )


# ── Action ──────────────────────────────────────────────────────────

class Action(BaseModel):
    model_config = {"use_enum_values": True}

    action_type: str
    target: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    justification: Optional[ActionJustification] = None



# ── Reward ──────────────────────────────────────────────────────────

class Reward(BaseModel):
    value: float = 0.0
    action_correctness: float = 0.0
    explanation_quality: float = 0.0
    reasoning_score: float = 0.0
    evidence_score: float = 0.0
    risk_accuracy_score: float = 0.0
    penalty: float = 0.0
    message: str = ""


# ── Grader Result ───────────────────────────────────────────────────

class TaskGraderResult(BaseModel):
    final_score: float = Field(ge=0.0, le=1.0)
    action_correctness: float = Field(ge=0.0, le=1.0)
    explanation_quality: float = Field(ge=0.0, le=1.0)
    threats_detected: int = 0
    threats_missed: int = 0
    containment_rate: float = 0.0
    eradication_rate: float = 0.0
    recovery_rate: float = 0.0
    steps_taken: int = 0
    feedback: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# EU AI Act Compliance Models
# ═══════════════════════════════════════════════════════════════════

class ComplianceCheck(BaseModel):
    """
    Individual compliance check result.
    Maps to specific EU AI Act articles.
    """
    check_name: str = Field(..., description="Name of the compliance check")
    passed: bool = Field(..., description="Whether this check passed")
    details: str = Field(..., description="Human-readable explanation of result")
    article_reference: str = Field(..., description="EU AI Act article reference")
    severity: str = Field(default="info", description="info|warning|critical")
    
    model_config = {"use_enum_values": True}


class AuditReport(BaseModel):
    """
    Complete EU AI Act Article 14 compliance audit report.
    
    Article 14: Human oversight for high-risk AI systems
    Article 13: Transparency obligations  
    Article 10: Data and data governance
    """
    report_id: str = Field(..., description="Unique audit report identifier")
    timestamp: str = Field(..., description="ISO-8601 timestamp of report generation")
    task_id: int = Field(..., description="Task ID that was audited")
    seed: int = Field(..., description="Random seed used for episode")
    episode_steps: int = Field(..., description="Total steps in episode")
    
    compliance_checks: list[ComplianceCheck] = Field(
        ..., description="List of individual compliance checks performed"
    )
    
    all_actions_justified: bool = Field(
        ..., description="True if all actions had adequate justifications (≥50 chars)"
    )
    explanation_quality_avg: float = Field(
        ..., ge=0.0, le=1.0, description="Average explanation quality score"
    )
    human_oversight_available: bool = Field(
        ..., description="True if human oversight mechanism (escalate) was available"
    )
    human_oversight_triggered: bool = Field(
        ..., description="True if agent escalated to human during episode"
    )
    high_risk_actions_count: int = Field(
        ..., ge=0, description="Count of high-risk actions (isolate/disable/restore)"
    )
    audit_trail_length: int = Field(
        ..., ge=0, description="Total number of actions in audit trail"
    )
    
    final_score: float = Field(
        ..., ge=0.0, le=1.0, description="Episode final score from grader"
    )
    compliant: bool = Field(
        ..., description="True if ≥4 out of 5 checks passed"
    )
    risk_level: str = Field(
        ..., description="Overall risk level: LOW|MEDIUM|HIGH|CRITICAL"
    )
    
    model_config = {"use_enum_values": True}
