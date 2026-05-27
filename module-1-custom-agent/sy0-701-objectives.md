# CompTIA Security+ SY0-701 Exam Objectives

**Exam Series:** SY0-701  
**Launch Date:** November 7, 2023  
**Estimated Retirement:** 2026  
**Number of Questions:** Maximum 90 (multiple-choice and performance-based)  
**Duration:** 90 minutes  
**Passing Score:** 750 (scale of 100–900)  
**Languages:** English, Japanese, Portuguese, Spanish, Thai  
**Recommended Experience:** CompTIA Network+ and two years in a security/systems administrator role  

---

## Domain Weightings

| Domain | Topic | Exam Weight |
|--------|-------|-------------|
| 1.0 | General Security Concepts | 12% |
| 2.0 | Threats, Vulnerabilities, and Mitigations | 22% |
| 3.0 | Security Architecture | 18% |
| 4.0 | Security Operations | 28% |
| 5.0 | Security Program Management and Oversight | 20% |

---

## Domain 1.0 — General Security Concepts (12%)

### 1.1 Security Controls
Compare and contrast types of security controls:
- **Technical** — hardware or software mechanisms (firewalls, encryption)
- **Managerial** — policies and procedures (AUP, separation of duties)
- **Operational** — day-to-day procedures carried out by people
- **Physical** — physical barriers and access controls

Control purposes:
- **Preventive** — stops an incident before it occurs
- **Detective** — identifies incidents after they occur
- **Corrective** — restores systems after an incident
- **Deterrent** — discourages attackers (e.g., warning signs, lighting)
- **Compensating** — alternative control when primary control is not feasible
- **Directive** — directs behavior through policy or regulation

### 1.2 Fundamental Security Concepts
- **CIA Triad:** Confidentiality, Integrity, Availability
- **Non-repudiation:** Proof that an action was performed by a specific party
- **AAA:** Authentication, Authorization, Accounting
- **Zero Trust:** Never trust, always verify — no implicit trust based on network location
- **Deception/Disruption Technology:** Honeypots, honeynets, DNS sinkholes

### 1.3 Change Management
- Business processes impacting security operations
- Technical implications of change
- Documentation and version control requirements

### 1.4 Cryptographic Solutions
- **PKI:** Public Key Infrastructure — certificates, CAs, trust chains
- **Encryption:** Symmetric (AES) and asymmetric (RSA) algorithms
- **Obfuscation:** Making data difficult to interpret
- **Hashing:** One-way function producing fixed-length output (SHA-256, MD5)
- **Digital Signatures:** Verify authenticity and integrity
- **Blockchain:** Distributed ledger providing tamper-evident records

---

## Domain 2.0 — Threats, Vulnerabilities, and Mitigations (22%)

### 2.1 Threat Actors and Motivations
Types of threat actors:
- **Nation-states** — advanced persistent threats, espionage
- **Unskilled attackers (script kiddies)** — use existing tools without deep knowledge
- **Hacktivists** — ideologically motivated
- **Insider threats** — malicious or negligent employees
- **Organized crime** — financially motivated
- **Shadow IT** — unsanctioned systems/services within an organization

Motivations: data exfiltration, espionage, financial gain, disruption, ideological

### 2.2 Threat Vectors and Attack Surfaces
- **Message-based:** Phishing, smishing, vishing
- **Unsecure networks:** Open Wi-Fi, rogue access points
- **Social engineering:** Pretexting, baiting, tailgating, impersonation
- **File-based:** Malicious attachments, macros
- **Voice call:** Vishing, caller ID spoofing
- **Supply chain:** Compromised vendors, hardware/software tampering
- **Vulnerable software:** Unpatched applications, zero-days

### 2.3 Vulnerabilities
- **Application:** Injection flaws, buffer overflows, XSS, CSRF
- **Hardware:** Firmware weaknesses, end-of-life components
- **Mobile device:** Sideloading, insecure app stores, jailbreaking
- **Virtualization:** VM escape, hypervisor attacks
- **OS-based:** Unpatched systems, misconfigurations
- **Cloud-specific:** Misconfigured storage buckets, insecure APIs
- **Web-based:** SQL injection, directory traversal
- **Supply chain:** Third-party component vulnerabilities

### 2.4 Malicious Activity (Indicators of Compromise)
- **Malware:** Ransomware, trojans, worms, rootkits, spyware, keyloggers
- **Password attacks:** Brute force, dictionary, credential stuffing, spraying
- **Application attacks:** Injection, privilege escalation, replay attacks
- **Physical attacks:** Skimming, card cloning, evil twin
- **Network attacks:** DDoS, on-path (MITM), DNS poisoning, ARP spoofing
- **Cryptographic attacks:** Downgrade, collision, birthday

IoC examples: abnormal outbound traffic, unexpected privilege escalation, 
unusual login times, hash value mismatches, unexpected patching/reboots

### 2.5 Mitigation Techniques
- **Segmentation:** VLANs, air gapping
- **Access control:** Least privilege, need-to-know
- **Configuration enforcement:** Baseline configs, hardening guides
- **Hardening:** Disabling unused services, closing ports
- **Isolation:** Sandboxing, quarantine
- **Patching:** Regular patch cycles, emergency patches

---

## Domain 3.0 — Security Architecture (18%)

### 3.1 Architecture Models
- **On-premises:** Organization owns and manages all infrastructure
- **Cloud:** SaaS, PaaS, IaaS — shared responsibility model applies
- **Virtualization:** VMs, containers, hypervisors
- **IoT:** Internet of Things — often legacy, limited patching capability
- **ICS/SCADA:** Industrial control systems — often air-gapped but increasingly connected
- **IaC:** Infrastructure as Code — security must be built into pipelines

### 3.2 Enterprise Infrastructure Security
- Security principles: defense in depth, least privilege, separation of duties
- Infrastructure considerations: redundancy, failover, secure communications
- Secure access: VPNs, jump servers, bastion hosts
- Network segmentation: DMZ for public-facing services, internal zones

### 3.3 Data Protection
Data states:
- **Data at rest:** Encrypted storage
- **Data in transit:** TLS, VPN
- **Data in use:** Memory encryption, secure enclaves

Data classifications: public, internal, confidential, restricted/top secret

Securing methods: tokenization, masking, obfuscation, rights management

### 3.4 Resilience and Recovery
- **High availability:** Redundant systems, load balancing, clustering
- **Site types:** Hot site (immediate failover), warm site (hours), cold site (days)
- **Backup strategies:** Full, incremental, differential; 3-2-1 rule
- **Testing:** Tabletop exercises, failover testing, simulations
- **Continuity of operations (COOP):** Plans for maintaining essential functions

---

## Domain 4.0 — Security Operations (28%)

### 4.1 Computing Resources
- **Secure baselines:** Standard configurations for systems
- **Mobile solutions:** MDM, MAM, BYOD policies, containerization
- **Hardening:** CIS benchmarks, STIG compliance
- **Wireless security:** WPA3, EAP, disabling WPS, rogue AP detection
- **Application security:** Input validation, secure coding, WAF
- **Sandboxing:** Isolating untrusted code/content
- **Monitoring:** SIEM, log aggregation, alerting

### 4.2 Asset Management
- Hardware asset tracking: acquisition through disposal
- Software asset management: licensing, version control
- Data asset classification and handling
- Secure disposal: degaussing, shredding, wiping (NIST 800-88)

### 4.3 Vulnerability Management
Process: Identify → Analyze → Remediate → Validate → Report
- Vulnerability scanning tools: Nessus, OpenVAS, Qualys
- CVSS scoring: Common Vulnerability Scoring System (0–10)
- CVE: Common Vulnerabilities and Exposures database
- Patch prioritization based on risk and criticality

### 4.4 Alerting and Monitoring
- **SIEM:** Security Information and Event Management
- **Log sources:** Firewall logs, authentication logs, application logs, netflow
- **Alerting:** Thresholds, anomaly detection, correlation rules
- **Monitoring tools:** IDS/IPS, EDR, packet analyzers

### 4.5 Enterprise Security Tools
- **Firewalls:** Stateful, NGFW, WAF
- **IDS/IPS:** Signature-based vs. anomaly-based detection
- **DNS filtering:** Blocking malicious domains
- **DLP:** Data Loss Prevention — prevent unauthorized data exfiltration
- **NAC:** Network Access Control — enforce policy before granting access
- **EDR/XDR:** Endpoint/Extended Detection and Response

### 4.6 Identity and Access Management
- **Provisioning/deprovisioning:** Onboarding and offboarding accounts
- **SSO:** Single Sign-On — one credential for multiple systems
- **MFA:** Multifactor Authentication — something you know/have/are
- **Privileged access:** PAM tools, just-in-time access, least privilege
- **Federation:** SAML, OAuth, OpenID Connect

### 4.7 Automation and Orchestration
- Use cases: automated patching, account provisioning, incident response playbooks
- Scripting benefits: consistency, speed, reduced human error
- Considerations: complexity, single point of failure, security of scripts

### 4.8 Incident Response
Process: Prepare → Identify → Contain → Eradicate → Recover → Lessons Learned
- **Training:** Tabletop exercises, red team/blue team
- **Root cause analysis:** Identifying how the incident occurred
- **Threat hunting:** Proactively searching for hidden threats
- **Digital forensics:** Chain of custody, order of volatility, preservation

### 4.9 Data Sources for Investigations
- Log data: authentication, application, firewall, DNS
- NetFlow: traffic metadata without full packet capture
- Packet captures: full content analysis (Wireshark)
- Threat intelligence feeds: IOC sharing, STIX/TAXII

---

## Domain 5.0 — Security Program Management and Oversight (20%)

### 5.1 Security Governance
- **Guidelines:** Recommended practices (not mandatory)
- **Policies:** Mandatory rules (AUP, password policy, data classification policy)
- **Standards:** Specific mandatory requirements (NIST, ISO 27001)
- **Procedures:** Step-by-step instructions for carrying out tasks
- **External considerations:** Laws, regulations, industry frameworks
- **Governance structures:** Board oversight, CISO role, security committees

### 5.2 Risk Management
- **Risk identification:** Assets, threats, vulnerabilities
- **Risk assessment:** Likelihood × Impact
- **Risk analysis:** Qualitative vs. quantitative
- **Risk register:** Documented list of identified risks
- **Risk tolerance/appetite:** How much risk the organization accepts
- **Risk strategies:** Accept, avoid, transfer, mitigate
- **BIA:** Business Impact Analysis — identifies critical functions and recovery priorities

### 5.3 Third-Party Risk Management
- Vendor assessment: security questionnaires, audits, penetration testing
- Vendor agreements: SLA, MOU, MSA, NDA, BPA
- Monitoring: ongoing vendor risk reviews
- Rules of engagement: defining acceptable testing scope

### 5.4 Security Compliance
- Compliance frameworks: PCI-DSS, HIPAA, GDPR, SOX, FERPA
- Reporting: internal and regulatory reporting requirements
- Consequences of non-compliance: fines, legal liability, reputational damage
- Privacy considerations: PII, PHI, data sovereignty

### 5.5 Audits and Assessments
- **Attestation:** Third-party confirmation of security posture
- **Internal audits:** Self-assessment against policy
- **External audits:** Independent third-party review
- **Penetration testing:** Simulated attacks to identify exploitable vulnerabilities
  - Types: black box, white box, gray box
  - Phases: reconnaissance, scanning, exploitation, post-exploitation, reporting

### 5.6 Security Awareness
- **Phishing training:** Simulated phishing campaigns, click rate tracking
- **Anomalous behavior recognition:** Training users to spot suspicious activity
- **User guidance:** Acceptable use, clean desk, social engineering defense
- **Reporting:** Clear channels for reporting suspected incidents
- **Monitoring:** Tracking training completion and effectiveness

---

## DoD 8140 Work Roles (Security+ Applicable)
Cyber defense analyst, incident responder, vulnerability analyst, security control assessor,
system administrator, network specialist, systems planner, IT project manager,
information security manager, secure software assessor

---

*Source: CompTIA Security+ SY0-701 Official Exam Objectives. Copyright © CompTIA, Inc.*  
*This file is for study and exam preparation purposes only.*
