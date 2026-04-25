"""
Network Topology Randomizer
==============================
Generates varied, realistic network environments with:
    - Multiple network segments (DMZ, Corporate, Production, etc.)
    - Randomized host counts and service configurations per segment
    - Inter-segment connectivity with realistic firewall rules
    - Vulnerability distribution scaled by difficulty
    - Seed-based reproducibility
"""
from __future__ import annotations
import random as _random
from typing import Any, Dict, List, Optional, Tuple

SEGMENT_DEFINITIONS = {
    "DMZ": {
        "roles": ["web-proxy", "reverse-proxy", "waf", "public-api"],
        "services": [["http", "https"], ["http", "https", "ssh"], ["https", "waf-mgmt"]],
        "criticality": "high",
        "exposure": "internet-facing",
    },
    "Corporate": {
        "roles": ["workstation", "mail-server", "file-share", "print-server", "vpn-gateway"],
        "services": [["rdp", "smb"], ["smtp", "imap", "https"], ["smb", "nfs"], ["ipp", "smb"]],
        "criticality": "medium",
        "exposure": "internal",
    },
    "Production": {
        "roles": ["app-server", "api-gateway", "load-balancer", "cache-server"],
        "services": [["http", "https", "ssh"], ["https", "grpc"], ["http", "https"], ["redis", "memcached"]],
        "criticality": "critical",
        "exposure": "internal",
    },
    "Development": {
        "roles": ["dev-workstation", "ci-runner", "staging-server", "git-server"],
        "services": [["ssh", "http"], ["ssh", "docker"], ["http", "ssh"], ["ssh", "git"]],
        "criticality": "low",
        "exposure": "internal",
    },
    "Database": {
        "roles": ["primary-db", "replica-db", "analytics-db", "cache-db"],
        "services": [["mysql", "ssh"], ["postgresql", "ssh"], ["clickhouse", "ssh"], ["redis"]],
        "criticality": "critical",
        "exposure": "restricted",
    },
    "IoT": {
        "roles": ["sensor-gateway", "camera-controller", "hvac-controller", "badge-reader"],
        "services": [["mqtt", "http"], ["rtsp", "http"], ["bacnet", "http"], ["http"]],
        "criticality": "low",
        "exposure": "isolated",
    },
    "SCADA": {
        "roles": ["plc-controller", "hmi-station", "historian-server", "eng-workstation"],
        "services": [["modbus", "opc-ua"], ["http", "rdp"], ["sql", "opc-ua"], ["rdp", "ssh"]],
        "criticality": "critical",
        "exposure": "air-gapped",
    },
    "Cloud": {
        "roles": ["k8s-node", "lambda-worker", "s3-bucket", "rds-instance"],
        "services": [["https", "kubelet"], ["https"], ["https", "s3"], ["mysql", "postgresql"]],
        "criticality": "high",
        "exposure": "cloud",
    },
}

# Inter-segment connectivity rules (realistic firewall topology)
SEGMENT_CONNECTIONS = {
    "DMZ": ["Corporate", "Production"],
    "Corporate": ["DMZ", "Production", "Development", "Database"],
    "Production": ["DMZ", "Corporate", "Database", "Cloud"],
    "Development": ["Corporate", "Cloud"],
    "Database": ["Corporate", "Production"],
    "IoT": ["Corporate"],
    "SCADA": [],  # Air-gapped by default
    "Cloud": ["Production", "Development"],
}

CVE_POOL = [
    "CVE-2024-3094", "CVE-2024-21626", "CVE-2023-44487", "CVE-2024-3400",
    "CVE-2024-21762", "CVE-2024-1709", "CVE-2023-46805", "CVE-2024-21887",
    "CVE-2024-27198", "CVE-2023-42793", "CVE-2024-29990", "CVE-2024-2961",
    "CVE-2024-20353", "CVE-2024-1708", "CVE-2023-4966", "CVE-2024-27199",
]


class NetworkTopologyGenerator:
    """
    Generates randomized but realistic network topologies.
    
    Complexity parameter controls:
        - Which segments are included
        - Number of hosts per segment
        - Vulnerability density
        - Inter-segment connection density
    """

    # Segments available at each complexity tier
    COMPLEXITY_TIERS = {
        0.0: ["Corporate", "Production"],
        0.3: ["Corporate", "Production", "DMZ"],
        0.5: ["Corporate", "Production", "DMZ", "Development", "Database"],
        0.7: ["Corporate", "Production", "DMZ", "Development", "Database", "Cloud"],
        0.9: list(SEGMENT_DEFINITIONS.keys()),  # All segments
    }

    def generate(
        self, seed: int, complexity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Generate a random network topology.

        Args:
            seed: Random seed for reproducibility
            complexity: 0.0-1.0 controls segments, hosts, vulnerabilities

        Returns:
            Dict with segments, hosts, connections, and graph
        """
        rng = _random.Random(seed)
        complexity = max(0.0, min(1.0, complexity))

        # Select segments based on complexity
        segments = self._select_segments(complexity)

        # Generate hosts per segment
        network: Dict[str, List[Dict]] = {}
        all_hosts: List[Dict] = []
        host_counter = 1

        for segment in segments:
            seg_def = SEGMENT_DEFINITIONS[segment]
            host_count = rng.randint(2, 2 + int(complexity * 8))
            segment_hosts = []

            for i in range(host_count):
                role = rng.choice(seg_def["roles"])
                services = rng.choice(seg_def["services"])
                vulns = self._assign_vulnerabilities(rng, complexity, segment)

                host = {
                    "id": f"{segment.lower()}-{i+1:02d}",
                    "host_id": f"HOST-{host_counter:03d}",
                    "hostname": f"{role}-{host_counter}",
                    "segment": segment,
                    "role": role,
                    "ip_address": self._generate_ip(rng, segment, i),
                    "services": list(services),
                    "criticality": seg_def["criticality"],
                    "exposure": seg_def["exposure"],
                    "vulnerabilities": vulns,
                    "connections": [],
                }
                segment_hosts.append(host)
                all_hosts.append(host)
                host_counter += 1

            network[segment] = segment_hosts

        # Create inter-segment connections
        graph = self._build_connections(rng, network, complexity)

        return {
            "segments": segments,
            "network": network,
            "all_hosts": all_hosts,
            "host_count": len(all_hosts),
            "segment_count": len(segments),
            "connections": graph,
            "complexity": complexity,
        }

    def _select_segments(self, complexity: float) -> List[str]:
        """Select network segments based on complexity level."""
        selected = ["Corporate", "Production"]  # Always present
        for threshold, segs in sorted(self.COMPLEXITY_TIERS.items()):
            if complexity >= threshold:
                selected = segs
        return list(selected)

    def _generate_ip(self, rng: _random.Random, segment: str, index: int) -> str:
        """Generate segment-appropriate IP addresses."""
        subnet_map = {
            "DMZ": "172.16.0", "Corporate": "10.0.1", "Production": "10.0.2",
            "Development": "10.0.3", "Database": "10.0.4", "IoT": "10.0.5",
            "SCADA": "10.0.6", "Cloud": "10.0.7",
        }
        subnet = subnet_map.get(segment, f"10.0.{rng.randint(8, 15)}")
        return f"{subnet}.{10 + index}"

    def _assign_vulnerabilities(
        self, rng: _random.Random, complexity: float, segment: str
    ) -> List[str]:
        """Assign vulnerabilities based on complexity and segment exposure."""
        # More exposed segments have more vulnerabilities
        exposure_factor = {"internet-facing": 1.5, "cloud": 1.3, "internal": 1.0,
                          "restricted": 0.7, "isolated": 0.5, "air-gapped": 0.3}
        seg_def = SEGMENT_DEFINITIONS.get(segment, {})
        factor = exposure_factor.get(seg_def.get("exposure", "internal"), 1.0)

        max_vulns = max(0, int(complexity * 3 * factor))
        if max_vulns == 0:
            return []

        num_vulns = rng.randint(0, max_vulns)
        return rng.sample(CVE_POOL, min(num_vulns, len(CVE_POOL)))

    def _build_connections(
        self, rng: _random.Random,
        network: Dict[str, List[Dict]],
        complexity: float,
    ) -> List[Dict[str, str]]:
        """Build inter-segment connections based on topology rules."""
        connections = []

        for segment, hosts in network.items():
            allowed_targets = SEGMENT_CONNECTIONS.get(segment, [])

            for host in hosts:
                # Intra-segment connections (always)
                peers = [h for h in hosts if h["id"] != host["id"]]
                if peers:
                    num_peers = rng.randint(1, min(3, len(peers)))
                    for peer in rng.sample(peers, num_peers):
                        conn = {"source": host["id"], "target": peer["id"], "type": "intra-segment"}
                        connections.append(conn)
                        host["connections"].append(peer["id"])

                # Inter-segment connections
                for target_seg in allowed_targets:
                    if target_seg not in network:
                        continue
                    # Connection probability scales with complexity
                    if rng.random() < 0.3 + complexity * 0.4:
                        target_host = rng.choice(network[target_seg])
                        conn = {"source": host["id"], "target": target_host["id"], "type": "inter-segment"}
                        connections.append(conn)
                        host["connections"].append(target_host["id"])

        return connections
