from typing import Dict

class ROICalculator:
    INDUSTRY_BENCHMARKS = {
        "avg_breach_cost_2024": 4_450_000,      # IBM Cost of a Data Breach Report
        "avg_time_to_detect_days": 204,
        "avg_time_to_contain_days": 73,
        "cost_per_day_breach_active": 22_000,
        "soc_analyst_salary": 95_000,
        "false_positive_cost_per_alert": 150,
        "avg_incidents_per_year": 45,           # Typical mid-large enterprise
    }
    
    def calculate_savings(self, agent_performance: Dict) -> Dict:
        """
        Calculate realistic ROI for deploying the AI SOC agent.
        """
        # 1. Faster Detection Savings
        agent_detection_steps = agent_performance.get("avg_detection_step", 12)
        steps_per_day = 4  # Assuming 6-hour work shifts in simulation
        agent_detection_days = agent_detection_steps / steps_per_day
        
        days_saved_per_incident = max(0, self.INDUSTRY_BENCHMARKS["avg_time_to_detect_days"] - agent_detection_days)
        detection_savings = days_saved_per_incident * self.INDUSTRY_BENCHMARKS["cost_per_day_breach_active"]
        
        # 2. Prevention Value
        prevention_rate = agent_performance.get("prevention_rate", 0.0)
        prevention_value = prevention_rate * self.INDUSTRY_BENCHMARKS["avg_breach_cost_2024"]
        
        # 3. False Positive Reduction
        baseline_fp_rate = 0.85
        agent_fp_rate = agent_performance.get("false_positive_rate", 0.4)
        total_alerts_per_year = 45_000  # Realistic annual alert volume
        
        fp_reduction = max(0, (baseline_fp_rate - agent_fp_rate) * total_alerts_per_year)
        fp_savings = fp_reduction * self.INDUSTRY_BENCHMARKS["false_positive_cost_per_alert"]
        
        # 4. Labor Savings
        hours_saved_per_alert = 0.4  # 24 minutes saved per alert
        annual_alerts = total_alerts_per_year
        hours_saved = annual_alerts * hours_saved_per_alert
        labor_savings = (hours_saved / 2080) * self.INDUSTRY_BENCHMARKS["soc_analyst_salary"]
        
        # 5. Total Annual Value
        total_annual_value = (
            detection_savings * self.INDUSTRY_BENCHMARKS["avg_incidents_per_year"] +
            prevention_value * self.INDUSTRY_BENCHMARKS["avg_incidents_per_year"] +
            fp_savings +
            labor_savings
        )
        
        # 6. Additional Credibility Metrics
        payback_period_months = max(1, 12000 / (total_annual_value / 12))  # Assuming $12K training + deployment cost
        
        return {
            "detection_speed_savings": round(detection_savings * self.INDUSTRY_BENCHMARKS["avg_incidents_per_year"]),
            "prevention_value": round(prevention_value * self.INDUSTRY_BENCHMARKS["avg_incidents_per_year"]),
            "false_positive_savings": round(fp_savings),
            "labor_cost_savings": round(labor_savings),
            "total_annual_value": round(total_annual_value),
            "formatted": f"${round(total_annual_value):,}",
            "payback_period_months": round(payback_period_months, 1),
            "roi_multiplier": round(total_annual_value / 12000, 1),  # vs $12K investment
            "assumptions": {
                "incidents_per_year": self.INDUSTRY_BENCHMARKS["avg_incidents_per_year"],
                "alerts_per_year": total_alerts_per_year
            }
        }