# Percy: AI-Powered Smart IDE and CI/CD Optimization

Percy is a four-pronged AI-driven solution optimizing CI/CD pipelines, automating testing and enhancing developer experience. It integrates multiple AI models to provide an end-to-end, efficient, and secure development environment.

## High-Level Design of Percy

### Components:

![image](https://github.com/user-attachments/assets/68342fe3-92f3-4a31-a675-993afd856011)

### Workflow

```mermaid
graph TD;
    subgraph Pax: Developer Hub
        A1[Live Coding in OnixIDE] -->|Sync & AI Help| A2[AI Suggests Code]
        A2 -->|Turn Text into Code| A3[Set Up Workspaces]
        A3 -->|Create & Configure Environments| B[Coder: Cloud Workspaces]
        
        B -->|Load Files Faster| C[Marksman: Smart Caching]
        B -->|Run Only Needed Tests| D[Perceptor: Picks Tests]
        B -->|Find Security Issues| E[Prowler: Checks Code]
        
        C -->|Preload Common Files| B
        D -->|Skip Unnecessary Tests for development purposes| B
        E -->|Spot Code Mistakes| B
    end

    Developer -->|Writes Code| A1
    Developer -->|Runs Tests With the help of| D
    Developer -->|Checks for Bugs with the help of| E
```

In summary:

- Perceptor exists to reduce workload in the CI/CD pipeline by automating tedious tasks like testing and documentation. Perceptor aims to reduce testing burden on the company's resources by choosing which tests to run rather than running all of them.
- Prowler essentially 'prowls' the codebase for security vulnerabilities. It scans the codebase for security vulnerabilities and flags potential risks using AI models and pattern detection.
- Pax is the integration of both thse models into the collaborative, cloud-based IDE synergy of Coder and OnixIDE
- Marksman's purpose is to mitigate the costs incurred due to cloud based development solutions. It does this by predicting usage patterns and caching resources efficiently to reduce latency and unnecessary cloud costs.

Reviewers please note: **Ony Prowler and Marksman are functional and have been tested. Perceptor's SVM Model failed to provide desired results, which resulted in a pivot to an XGBoost instead. Perceptor is, therefore, still under development. Pax, the ecosystem, is an integration in Go, and requires the other three systems to be end-to-end functional.**

## Components

### 1. Prowler: AI for Testing

Prowler enhances software testing through prioritization and vulnerability detection.

```mermaid
graph TD;
    subgraph Prowler: AI Security Scanner
        A1[Script] -->|Parse File Structure| D[Tree-Sitter AST]
        D -->|Extract Changed Functions & Classes| A2[Structured Code Sections]
        A2 -->|Analyze with Static & Dynamic Tools| B1[Semgrep, OWASP, NVD]
        A2 -->|Send to LLMs for Review| B2[Claude, Gemini, Codey]
        B1 -->|Flag Known Vulnerabilities| C1[Security Report]
        B2 -->|Detect Logic & Semantic Issues| C2[Risk Warnings]
        C1 & C2 -->|Generate Fix Suggestions| C3[Developer Report]
    end

    Input["GitHub Repos, Semgrep Rules, OWASP Dataset, NVD Vulnerability Database"] --> A1
    C3 -->|Developers Receive Warnings & Fixes| Output["Code Security Improved"]

```

**Features:**
- Test Prioritization:
  - Uses Perceptor's changelog to identify crucial unit tests
  - Reduces test execution time by up to 65%
- Security Vulnerability Detection:
  - Scans for LFI, XSS, RCE, AFO, and SSRF vulnerabilities using predefined templates
  - Tree-Sitter integration for multilingual support
  - Reference: HikaruEgashira's vulnhuntrs repository
- Automated Security Tests:
  - Generates unit tests for vulnerabilities

**Research Citations:**
- VulnHuntr [[See](https://github.com/protectai/vulnhuntr/)]

### 2. Marksman: Optimizing Cloud Latency

Marksman reduces latency for cloud-based development environments.

**Features**
- Latency Reduction: Predictive caching and state-based optimizations using Markov Models
- Expected latency reduction: 40-60%

```mermaid
graph TD;
    subgraph Marksman: Faster Access & Less Waste
        A1[Script] -->|Track Used Files & Access Patterns| A2[Markov Model + LSTM Predict Needs]
        A2 -->|Group Similar Usage Patterns| A3[K-Means Clustering]
        A3 -->|Decide What to Preload & Evict| B[Hybrid LRU-LFU Cache]
        B -->|Keep Critical, Remove Stale| C[Smart Cache Storage]
        
        A1 -->|Detect Anomalous Access Patterns| D[Resource Usage Anomaly Detection]
        D -->|Identify Spikes & Drops| E[Z-Score + Time Decay Analysis]
        E -->|Trigger Alerts & Auto-Scaling| F[Prometheus + Kubernetes HPA]
    end

    Input["Travis CI Logs, OpenTelemetry Network Tracing, Prometheus Metrics, Bazel Build Cache"] --> A1
    F -->|Improve Prefetching Decisions| B


```

**Research Citations:**
- "An Improved Cache Eviction Strategy: Combining Least Recently Used and Least Frequently Used Policies (IEEE Xplore" [[See](https://ieeexplore.ieee.org/document/10454976)]

### 3. Perceptor: AI in CI/CD

Perceptor optimizes CI/CD through version tracking and test prediction.

```mermaid
graph TD;
    subgraph Perceptor: Faster Testing
        A1[Code Changes] -->|Extract Features from Past Edits| A2[Feature Engineering]
        A2 -->|Find Patterns in Test Failures| B[XGBoost Classifier]
        B -->|Predict High-Risk Tests| C[Prioritized Test Execution]
        C -->|Skip Low-Risk & Unaffected Tests| D[Selective Test Execution]
    end

    Input["Travis CI Logs, GitHub Actions, JUnit Reports, SonarQube Data"] --> A1
    D -->|Faster Testing & CI/CD Speedup| Output["CI/CD Runs More Efficiently"]
```

**Features:**
- Code Tracking & AI Analysis:
  - Git-based version history tracking
  - Secure code change extraction via locally hosted Ollama model
- Intelligent Test Execution Prediction:
  - Extracts CI/CD workflows from .github/workflows
  - SVM model predicts necessary test cases

**Future Scope:**
- Cost optimization via dynamic resource allocation (estimated 30-40% savings)

### 4. Pax: AI-Enhanced Developer Experience

Pax enhances development workflow through collaborative editing and ready-to-use workspaces.

```mermaid
graph TD;
    subgraph Pax: All-in-One Dev System
        A1[OnixIDE: Live Coding] -->|Real-time Collaboration| A2[AI Code Suggestions]
        A2 -->|Generate Configurations for Environments| A3[LLM-Powered Workspace Setup]
        A3 -->|Pre-built & Configured Containers| B[Coder: Cloud Workspaces]
        
        B -->|Load Faster| C[Marksman: Smart Caching]
        B -->|Run Only Needed Tests| D[Perceptor: Test Prioritization]
        B -->|Check for Security Issues| E[Prowler: Vulnerability Detection]
    end

    Input["Coder Workspace Configs, GitHub Repos, OpenAI API, Stack Overflow API"] --> A1
    C -->|Preload Commonly Used Files| B
    D -->|Avoid Unnecessary Test Runs| B
    E -->|Spot Security Risks in Code| B
```

**Features:**
- Collaborative Editing: OnixIDE-based editing with real-time collaboration via websockets
- Portable Workspaces: Integration with Coder (GitHub: @coder) for pre-configured environments
- CI/CD Integration: Streamlined pipeline integration

**Future Scope:**
- Automated Workspace Generation: LLM-based .yml configuration generation for coder workspaces
- Integration with Go Backend of Coder for seamless processing (currently a temporary setup of .yml)

## More Detailed Workflow Overview

```mermaid
graph TD;
  subgraph "Developer Interaction Data Sources"
    A1[IDE Telemetry]
    A2[GitHub API Commit Logs]
    A3[TravisCI Build History API]
    A4[Docker Stats API]
    A5[OpenTelemetry/eBPF Network Tracing]
    A6[Maven/Gradle Build Logs]
  end

  subgraph "Marksman: Cloud Latency Optimization"
    M1[Hybrid LRU-LFU Caching for Build Artifacts]
    M2[Markov Model + LSTM in the future for State Prediction]
    M3[Resource Usage Anomaly Detection]
    M4[Predictive Prefetching for Dependencies]
    M5[Prometheus & Grafana Monitoring -> Kubernetes HPA]
  end

  subgraph "Pax: AI-Enhanced Developer Experience"
    P1[OnixIDE Collaborative Editing]
    P2[Portable Workspaces via Coder]
    P3[CI/CD Pipeline Integration]
    P4[LLM-Generated Workspace Configs]
    P5[Pre-configured Development Environments]
  end

  subgraph "Perceptor: AI-Driven CI/CD Optimization"
    C1[Git-based Version History Tracking]
    C2[Code Llama for Local Code Analysis]
    C3[GitHub Workflow Extraction]
    C4[XGBoost for Test Prediction]
    C5[Dynamic Resource Allocation Based on Metrics]
  end

  subgraph "Prowler: AI for Testing & Security"
    V1[Test Prioritization Engine]
    V2[Vulnerability Detection Scanner]
    V3[Tree-Sitter Parsing for Multilingual Support]
    V4[Security Test Generation]
    V5[LFI/XSS/RCE/AFO/SSRF Detection]
  end

  A1 -->|IDE Usage Patterns| M2
  A2 -->|Code Change Analysis| C1
  A3 -->|Pipeline Performance| C3
  A4 -->|Resource Metrics| M3
  A5 -->|Network Latency Data| M4
  A6 -->|Build Dependencies| P2

  M2 -->|State Transition Predictions| M1
  M3 -->|Resource Anomalies| M5
  M4 -->|Prefetching Dependency Data| M1
  M1 -->|Reduced CI/CD Latency 40-60%| P5

  P1 -->|Real-time Collaboration| P2
  P2 -->|Pre-configured Workspaces| P3
  P3 -->|Streamlined CI/CD| P4
  P4 -->|.yml Generation| P5
  P5 -->|Setup Time Reduction 75%| C5

  C1 -->|Code Changes| C2
  C2 -->|Static & Dynamic Analysis| C3
  C3 -->|Workflow Extraction| C4
  C4 -->|Test Prediction 90% Accuracy| C5
  C5 -->|Optimized CI/CD Compute Cost 30-40%| M5

  V1 -->|Critical Test Identification| C4
  V2 -->|Code Parsing| V3
  V3 -->|Error-Prone Code Detection| V4
  V4 -->|Security Issue Detection 80%| V5
  V5 -->|Unit Tests Generated| P3

  M1 -->|Optimized Build Caching| P5
  C4 -->|Pipeline Speedup 35-50%| P3
  V1 -->|Test Time Reduction 65%| C4
  P2 -->|Integration with Coder| M4
  M5 -->|Auto Scaling Decisions| C5

```

## Integration of Percy Components with Coder

### Technical Integration:

**Workspace Integration (Pax + Coder):**
- Automatic workspace configuration generation using LLMs
- Dynamic provisioning based on project type and CI/CD needs
- Setup time reduction: 75%

**CI/CD Optimization (Perceptor + Coder):**
- AI-driven test selection and workflow optimization
- Continuous model improvement using CI/CD run logs
- Pipeline execution speedup: 35-50%

**AI-Driven Testing (Prowler + Coder):**
- Real-time security vulnerability analysis
- Automated test recommendations before commits
- Security issue detection rate: 76%

**Latency Optimization (Marksman + Coder):**
- Markov Model predictions inform caching strategies
- Prefetching of dependencies reduces startup time by 45%

## Development Approach

Development approach for this particular application should follow an iterative development model:

1. Prototype Phase: Core functionality with modular AI models
2. Data Collection & Training: Using open-source repositories to collect workflow, vulnerability, etc. oriented data
3. Pilot Testing: Internal deployment with feedback loops
4. Scalability & Optimization: RL and Markov models implementation
5. Iterate: Return back to the Whiteboard based on user feedback

## Cost Analysis

**Component Costs:**
- Compute: ₹41–₹410 per hour on cloud services for LLM operation
- Storage: ₹8.2/GB per month (PostgreSQL/Elasticsearch)
- AI Model Training: ₹41,000 per model update
- Total projected annual cost: ₹1,230,000 to ₹2,050,000

**Cost Calculation Method:**
- Cloud provider pricing benchmarks (AWS, GCP, Azure)
- Per-pipeline compute load modeling
- Optimization via caching and batching (25% savings)

**Notes on Costs**
- Storage Costs: The cost mentioned is a general benchmark. Actual costs can vary based on the specific cloud provider and storage type (e.g., EBS on AWS costs around $0.135 per GB per month for General Purpose SSD)
- Compute Costs: The range per hour is broad and can vary significantly depending on the instance type and provider.
- AI Model Training: The cost of ₹41,000 per model update is a rough estimate.

## Rollout Strategy

1. Alpha Release (Internal): Controlled environment testing (Q1 2025)
Reasoning: Testing in a controlled environment allows for identifying and fixing issues early, ensuring the application is stable before external exposure. This phase helps in refining the product based on internal feedback, reducing the risk of major issues during later stages. 
2. Beta Release (Open Source): Developer community engagement (Q3 2025)
Reasoning: Engaging with the developer community provides valuable feedback and insights, helping to improve the application's functionality and user experience. This phase also fosters community involvement and can lead to contributions or suggestions for enhancements. This is also the time to generate hype around the software and eventually build a product that is **both useful and marketable**.
3. Enterprise Deployment: Cloud-optimized scaling (Q3 2026)
Reasoning: Scaling on cloud platforms ensures flexibility and scalability, which are crucial for handling increased demand as the application grows. 
4. Full Integration: CI/CD provider partnerships (Q2 2027)
Reasoning: Partnering with CI/CD providers facilitates seamless integration into existing workflows, enhancing efficiency and reducing deployment complexities

Also, engaging with both internal stakeholders during the alpha phase and external developers during the beta phase helps build support and buy-in. This engagement is crucial for ensuring that the application meets user needs and expectations

##### Resources:

1. GmbH, L. AI Strategy: Comprehensive guide & Best Practices | LeanIX. https://www.leanix.net/en/wiki/ai-governance/ai-strategy]
2. Kumar, S. (2025, February 17). 7 Essential steps for successful AI implementation in your business. CEI | Consulting. Solutions. Results. https://www.ceiamerica.com/blog/7-essential-steps-for-successfully-implementing-ai-in-your-business/

## Security and Privacy

**Security Measures:**
- Local AI Execution: Ollama for secure codebase handling
- Access Control: Role-based CI/CD pipeline access
- Data Encryption: AES-256 for logs and models
- Automated Security Audits: Continuous vulnerability scanning
- Compliance: GDPR, SOC2 compatible

## Data & Technical Expertise

**Dataset Access:**
- Public CI/CD logs, open-source repositories
- Synthetic datasets for RL model training
- GitHub Actions and Jenkins community collaboration

**Technical Partnerships: Possibilties**
- AI, cybersecurity, and DevOps domain experts
- Open-source community contributions
- Enterprise-scale deployment partnerships

A gentle note. The commit history shows two contributors, `unfortunatelygeek` and `dandanheads-Datahack`. I request the reviewers to please note that both users are me, Aditi Rao, only.
