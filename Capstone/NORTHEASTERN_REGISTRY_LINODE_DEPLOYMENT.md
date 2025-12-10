

# 🚀 **Northeastern Registry – Linode Deployment Guide**

**Production Deployment for NANDA Registry + AgentFacts API**
High-availability agent discovery, metadata storage, and MCP integration powered by FastAPI + MongoDB Atlas.

---

## 📌 **Overview**

This repository contains the deployment script and instructions for running the **NANDA Registry** and **AgentFacts** services on a secure Linode instance.

The deployment provides:

* 🔎 **Agent Registration & Discovery**
* 📚 **AgentFacts metadata storage**
* 🤖 **MCP Server compatibility** (Model Context Protocol)
* 🌐 **REST APIs for agents & external systems**
* 💾 **Persistent MongoDB Atlas storage**
* 🔐 **Secure Linode firewall + SSH setup**

---

## 🧱 **Architecture**

```mermaid
flowchart LR
    A[NANDA Agent] -->|Registers| B[Registry API :6900]
    B -->|Reads/Writes| D[(MongoDB Atlas)]
    C[AgentFacts API :8000] -->|Reads/Writes| D
    E[MCP Server :9090] -->|Uses Registry Tools| B

    subgraph Linode VM
        B
        C
        E
    end
```

---

## 🛠 **Features**

### ✔ Agent Registry

Register, list, update, delete, and discover agents.

### ✔ AgentFacts

Rich metadata describing agent skills, performance, certifications.

### ✔ MCP Server Integration

Expose registry tools to LLMs (Claude, etc.) via MCP.

### ✔ Automated Deployment Script

Provision VM, set firewall, upload files, and configure services in minutes.

---

## 📦 **Deployment Script**

Use this script to provision everything automatically:

```
linode_deploy_registry_only.sh
```

It performs:

* Linode instance creation
* Ubuntu environment setup
* Firewall configuration
* SSH key provisioning
* Virtualenv creation
* Dependency installation
* Supervisor service configuration
* Automatic service startup

---

## 🚀 Quick Start

### 1️⃣ **Run the Deployment Script**

```bash
bash linode_deploy_registry_only.sh "<MONGODB_ATLAS_URL>"
```

Example:

```bash
bash linode_deploy_registry_only.sh \
"mongodb+srv://nanda_admin:password@cluster.mongodb.net/?retryWrites=true&w=majority"
```

---

## 🌐 Exposed Services

| Service                  | Port   | Description                           |
| ------------------------ | ------ | ------------------------------------- |
| **Registry API**         | `6900` | Agent registration, lookup, discovery |
| **AgentFacts API**       | `8000` | Metadata retrieval                    |
| **Static UI (optional)** | `9000` | Registry Dashboard                    |
| **MCP Server**           | `9090` | Agent discovery via MCP               |

---

## 📁 Project Structure (on Linode)

```
/home/ubuntu/nanda-registry/
    ├── venv/
    ├── src/
    │   ├── agentIndex.py          # Registry API
    │   ├── agentFactsServer.py    # AgentFacts API
    │   └── agent_mcp.py           # MCP Server
    ├── static/                    # Optional UI files
    └── supervisor/
```

---

## 🧩 API Health Checks

```bash
curl http://<IP>:6900/health
curl http://<IP>:8000/health
curl http://<IP>:9090/health
```

Expected:

```json
{"status":"healthy","mongodb":"connected"}
```

---

## 🧑‍💻 SSH Access

```bash
ssh -i nanda-registry-key root@<LINODE_PUBLIC_IP>
```

---

## 🛠 Supervisor Service Management

Check running services:

```bash
supervisorctl status
```

Restart a service:

```bash
supervisorctl restart registry
supervisorctl restart agentfacts
supervisorctl restart mcp
```

---

## 📝 Agent Registration Example

```bash
curl -X POST http://<IP>:6900/register \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "data-scientist-01",
    "agent_url": "http://<AGENT_IP>:8080",
    "capabilities": ["ml", "analysis"],
    "domain": "data_science",
    "description": "ML agent for model training"
  }'
```

---

## 🗂 Serving an Optional UI on Port 9000

Copy a file:

```bash
scp -i nanda-registry-key registry_ui.html root@<IP>:/root/
```

Create Supervisor service:

```bash
supervisorctl reread
supervisorctl update
supervisorctl restart ui_server
```

Open in browser:

```
http://<IP>:9000/registry_ui.html
```

---

## 🛑 Troubleshooting

### ❌ MongoDB connection errors

Ensure:

* Valid Atlas URL
* Atlas network access includes Linode IP
* Correct user/password

### ❌ Supervisor not running

Run:

```bash
sudo systemctl restart supervisor
```

### ❌ Port blocked

Verify firewall rules:

```bash
linode-cli firewalls view nanda-registry-only
```

---

## 🧹 Tear Down Deployment

Delete Linode instance:

```bash
linode-cli linodes delete <INSTANCE_ID>
```

Delete firewall:

```bash
linode-cli firewalls delete <FIREWALL_ID>
```

---

## ⭐ Final Notes

* This deployment is **production-ready** and follows best practices.
* Works seamlessly with ** AI Agents**, **Claude MCP**, and **AgentFacts**.
* Tested on **Linode Ubuntu 22.04 (g6-standard-1)**.


