AnomalyGuard — Project Blog Post
I Built an AI That Thinks Like a Cybersecurity Analyst — Here Is What I Learned

- By VSSK Sri Padmavathi | Solo Participant | Meta PyTorch OpenEnv HuggingFace X Scaler Hackathon 2026

Why I Started Building This ?

We live in a world where AI is everywhere.
In our phones. In our hospitals. In our banks. In the systems that run power grids and water supplies and financial markets. And wherever AI goes, attackers follow.
I started noticing this pattern about a year ago. Every week there was a new breach. Every month a new attack that exploited some system that was never designed to be a target. The scale of it was overwhelming. Security teams were drowning in alerts they could not process fast enough.
I started thinking about this problem seriously. Not as a news story. As an engineering challenge.
Then the Vercel breach happened.
When I read about it, something clicked for me. Here was a company with serious engineering talent, serious security investment, and it still happened. Not because they were careless. But because the attackers were fast, the alerts were noisy, and the window to respond was narrow.
I thought — the problem is not that security teams do not know what to do. The problem is that there are too many alerts, too little time, and too much noise to filter through manually.

What if an AI could help with the investigation?

Not just flag alerts. Actually investigate them. Check the host. Correlate the evidence. Explain what it found. And justify every action it took in a way a human analyst could review and trust.
That is what I set out to build. I wanted my environment to be strong enough that ML models trained on it could respond to realistic attacks — the kind of attacks that are actually happening right now, not toy examples from a textbook. That became AnomalyGuard.

What the Agent Actually Does?

Imagine you are a security analyst. It is 2am. Your SIEM has fired three alerts.

- Alert one says there is a C2 beacon going out to an IP in Eastern Europe every 300 seconds.
- Alert two says someone failed to log in via SSH fifty times in the last hour.
- Alert three says a scheduled task was created on your web server.

Which one do you investigate first?

How do you know the C2 beacon is not a false positive?
Before you isolate that web server do you know
for certain it is actually compromised?
This is exactly what AnomalyGuard trains an AI to do. The agent starts each episode knowing almost nothing. It can see that hosts exist on the network. But it cannot see whether they are compromised. The dangerous information — whether a host has active command and control, whether it has a persistence mechanism, what vulnerabilities it has — is hidden.
The agent has to earn that information. It has to call query_host and investigate before it can act.
If it tries to isolate a host without investigating it gets penalized. If it ignores a critical alert and just monitors instead of acting it gets penalized. If its reasoning is vague and uncited it gets penalized.
But when it investigates properly, cites specific evidence, explains why it chose this action over alternatives, and documents its risk assessment — that is when it gets rewarded.

The Night I Realized the Reward Function Was Wrong

When I ran my first training run and watched the reward go up. I felt good. The numbers were rising. I thought the agent was learning.
Then I looked at what the agent was actually doing. It had learned to output perfectly formatted JSON. But every single response said monitor. Just monitor. Do nothing. Collect no evidence. Take no action.
The reward was going up because the format was correct. But the agent had found the easiest path — do nothing, write it up nicely, collect the reward.
That is called reward hacking. And it taught me something important. The reward function is not just a number. It is your entire specification of what good
behavior looks like. If you get it wrong you do not get a bad agent. You get an agent that is very good at gaming the thing you built.
I redesigned the reward from scratch. Format compliance alone would not score well. The agent had to choose the right action type. The reasoning had to be long enough and contain actual security keywords. It had to cite evidence. It had to include a risk assessment with confidence levels. It had to consider alternatives. And critically I added an AntiHackingGuard that detects repetitive patterns and penalizes them. If the agent tries to just say monitor every turn the guard catches it and reduces the score.
After that redesign the training curves started telling a real story.

The Architecture That Made It Work
AnomalyGuard is not a single agent acting alone. Real security operations centers do not work that way. They have a triage analyst who classifies alerts, a containment specialist who isolates systems and a forensics expert who collects evidence and oversees recovery.
So I built three agents.
The Triage Agent sees alerts first and decides which are real threats and which are false positives. It has to be right because its decisions gate everything that comes after.
The Containment Agent takes over once threats are confirmed. It isolates hosts, blocks malicious IPs, disables compromised accounts. But it can only act
on hosts that have been investigated — the Triage Agent has to have queried them first.
The Forensics Agent handles eradication and recovery. It removes persistence mechanisms, patches vulnerabilities, and restores clean systems to production. But it can only restore a host after the Containment Agent has isolated it and the persistence has been cleared.
This creates a chain of dependencies that mirrors real incident response workflow. The agents have to coordinate. An agent that acts out of order gets penalized. An agent that skips a phase cannot complete the episode successfully.
What emerged from training was something I did not fully anticipate. The agents started to develop handoff behavior. The Triage Agent learned to flag
specific hosts that needed investigation. The Containment Agent learned to wait for that signal before acting. The Forensics Agent learned the sequence of steps required before restoration was even possible.
This is what the Multi-Agent Interactions theme is really about. Not just multiple agents running in parallel. Agents that model each other and coordinate toward a shared goal.

Making the Scenarios Realistic
This was the part I cared about most. Because my whole reason for building this was that I wanted models to respond to realistic attacks. Not toy examples. Not abstract grid worlds. Real attack patterns that are happening right now in the real world.
AnomalyGuard uses MITRE ATT&CK technique chains. Every scenario is generated from real attack archetypes:
C2 beaconing where a compromised host contacts a command and control server every few minutes.
The IP 185.220.101.45 that appears in training is a real known malicious IP linked to ransomware campaigns.
Credential dumping where an attacker accesses LSASS memory to steal passwords and move laterally across the network. MITRE technique T1003.
Log4Shell exploitation using CVE-2021-44228. The exact vulnerability that caused widespread damage when it was discovered in 2021 and is still being exploited today.
Lateral movement through compromised admin accounts. Persistence mechanisms hidden as scheduled tasks. Data exfiltration through DNS tunneling.
Every scenario the agent trains on is grounded in techniques that real attackers use against real organizations.

When I read about the Vercel breach I thought —
my environment needs to be strong enough that a model trained on it would have a chance of detecting and responding to something like that. That is the standard I held myself to.

The Curriculum That Teaches Itself

One of my favorite parts of AnomalyGuard is the adaptive curriculum.
When training begins the environment starts at level one. Simple scenarios. Alert triage only.
A few hosts. Clear signals.

As the agent improves the curriculum advances. Level four adds containment tasks. Level seven adds the full incident response lifecycle —
detection, containment, eradication, recovery — all in a single episode with up to thirty steps.
But here is what makes it different from a fixed curriculum.
The environment watches the agent's performance. If the agent is succeeding more than seventy-five percent of the time the difficulty increases. If it drops below thirty-five percent the difficulty decreases. The curriculum finds the edge of the agent's ability and keeps training there.
The training steps are dynamic too. A beginner scenario runs for fifty steps. An expert scenario runs for two hundred or more. The system decides how long to train based on
how complex the current task is. This is the Self-Improvement theme made concrete. The environment does not just provide a fixed challenge. It actively drives the agent's capability growth by adjusting difficulty and training duration based on what the agent can currently do.

What the EU AI Act Forced Me to Build Better?

When I decided to make EU AI Act compliancecentral to the reward function I thought it
would be a nice extra feature. It turned out to be the thing that made
the whole system more rigorous. Every action the agent takes is evaluated against
five compliance checks:
Article 14.4(b) — Did the agent justify its action with at least fifty characters of specific reasoning?
Article 13.1 — Is the explanation quality above0 the minimum threshold for transparency?
Article 14.1 — Is a human escalation option always available and accessible?
Article 14.4(c) — Are high-risk actions like isolation and account disabling properly documented?
Article 10.2(f) — Is the agent classifying alerts in a balanced way without systematic bias?
When I started enforcing these at the reward levelsomething unexpected happened. The agent's overall performance improved. Not just on the compliance metrics. On the actual security task.
It turns out that forcing the agent to explain its reasoning, cite evidence, and consider alternatives made it better at the underlying investigation. When you have to justify why you are isolating a host you think more carefully about whether you should.
The compliance requirement was not a constraint on good behavior. It was a driver of it.

Training Results

After running GRPO training with dynamic step
selection based on curriculum level here is
what the model learned.

![Training Progress](training_progress.png)

| Metric            | Before Training | After Training |
| ----------------- | --------------- | -------------- |
| Total Reward      | 0.792           | 0.953          |
| JSON Format       | 1.000           | 1.000          |
| Action Type       | 0.750           | 0.950          |
| Reasoning Quality | 0.750           | 0.920          |
| Evidence Citation | 0.750           | 0.950          |
| Risk Assessment   | 0.750           | 0.950          |

The reward curve shows the model moving from uncertain poorly-justified responses toward consistent evidence-based actions with proper EU AI Act compliant reasoning.

What I Would Tell Someone Starting This

If you are thinking about building an RL environment for LLM training here is what I wish someone had told me at the start.
The environment is harder than the model. Everyone talks about which model to use, which training algorithm, which hyperparameters. But the thing that will determine whether your agent actually learns something useful is the quality of your environment and your reward function.
Spend most of your time there.
Also inspect your agent's actual outputs during training. Do not just watch the reward curve. Look at what the model is actually saying. Look at whether it is genuinely investigating
or just gaming the format.
The reward curve will lie to you if you let it. The actual outputs never do.
And finally — make your scenarios realistic.
The reason I built AnomalyGuard was that I wanted models to respond to the kinds of attacks that are actually happening. Not toy problems. Real attack chains. Real IOCs. Real MITRE techniques.
Because the Vercel breach was real. And the next one will be too.

Try It Yourself

The environment is live. You can interact with it
right now.

Start an investigation:
https://padmavathi-123-anomalyguard.hf.space

Read the full API:
https://padmavathi-123-anomalyguard.hf.space/docs

Source code:
https://github.com/Padmavathi-1234/Anomaly-Guard-

Try investigating an incident yourself.
See how many steps it takes you to correctly
identify and contain the threat.

Then imagine doing that two thousand times a day.

That is why this matters.

VSSK Sri Padmavathi
Meta PyTorch OpenEnv HuggingFace X Scaler Hackathon 2026
Themes: World Modeling, Multi-Agent Interactions, Self-Improvement
