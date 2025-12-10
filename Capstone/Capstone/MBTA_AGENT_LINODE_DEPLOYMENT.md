

---

# 🚇 **MBTA Multi-Agent Deployment on Linode**

*Automated deployment of Alerts, Planner, StopFinder, Chat Backend & NANDA A2A Wrapper*

![Linode](https://img.shields.io/badge/Cloud-Linode-00A95C?style=flat\&logo=linode\&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat\&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Agents-009688?style=flat\&logo=fastapi)
![NANDA](https://img.shields.io/badge/NANDA-A2A%20Agents-purple?style=flat)

This repository contains the deployment script to provision and run the **MBTA Multi-Agent System** on a secure Linode instance.
The stack includes:

* 🚨 **Alerts Agent**
* 🗺️ **Route Planner Agent**
* 🔍 **Stop Finder Agent**
* 💬 **Chat Backend**
* 🤖 **NANDA A2A Adapter + Wrapper**
* 🔐 Firewall + Supervisor-managed services

Everything is deployed and configured automatically via:

```
linode-deploy-mbta-agent-only.sh
```



# 🧰 1. Purpose

This script deploys the complete **MBTA Multi-Agent Stack** onto a single Linode instance and connects it to an existing NANDA Registry.
The system includes:

* Fully container-less installation (raw Python + venv)
* A2A communication between agents and the NANDA ecosystem
* Chat UI to interact with MBTA agents
* Supervisor-managed agent services

It automatically provisions:

* ✔ Linode instance
* ✔ Firewall
* ✔ SSH setup
* ✔ Python + venv
* ✔ All agents + Chat backend
* ✔ A2A Adapter + Wrapper
* ✔ Service manager (Supervisor)

📝 *Every deployment run creates a brand-new instance with a unique Deployment ID.*

---

# 🧾 2. Requirements

## 2.1 Linode / CLI

* Active Linode account
* Linode CLI installed

```bash
linode-cli configure
linode-cli --version
```

## 2.2 Local Environment

You must have:

* Bash (macOS, Linux, WSL, or Git Bash)
* `ssh`, `scp`, `ssh-keygen`
* `openssl`
* Local MBTA agent project containing:

```
agents/alerts/main.py
agents/planner/main.py
agents/stopfinder/main.py
server/app.py              # Chat backend
requirements.txt
linode-deploy-mbta-agent-only.sh
```

---

# 🔑 3. Required Inputs

The script requires:

| Argument               | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| **MBTA_API_KEY**       | Your MBTA API key                                             |
| **LOCAL_PROJECT_PATH** | Path to your MBTA project                                     |
| **REGISTRY_URL**       | URL of your NANDA Registry (e.g., `http://45.79.123.45:6900`) |

Optional:

| Argument      | Default                 |
| ------------- | ----------------------- |
| REGION        | `us-east`               |
| INSTANCE_TYPE | `g6-standard-2`         |
| ROOT_PASSWORD | Generated automatically |

---

# 🚀 4. Script Usage

## 4.1 Linux/macOS/WSL

```bash
bash linode-deploy-mbta-agent-only.sh \
  <MBTA_API_KEY> \
  <LOCAL_PROJECT_PATH> \
  <REGISTRY_URL>
```

Example:

```bash
bash linode-deploy-mbta-agent-only.sh \
  mbta-XXXXXXXXXXXXXXXXXXXX \
  . \
  http://45.79.123.45:6900
```

## 4.2 Windows (Git Bash)

From PowerShell using Git Bash:

```powershell
"C:/Program Files/Git/bin/bash.exe" `
  "C:/path/linode-deploy-mbta-agent-only.sh" `
  "mbta-xxxxxxxxxxxxxxxxxxxx" `
  "C:/myproject" `
  "http://45.79.123.45:6900"
```

If already inside Git Bash:

```bash
bash linode-deploy-mbta-agent-only.sh \
  "mbta-xxxxxxxxxxxxxxxxxxxx" \
  "." \
  "http://registry_ip:6900"
```

---

# ⚙️ 5. What the Script Does

The script performs **full server provisioning**:

---

### 🧭 Local Machine

* Validates inputs
* Packages project into `.tar.gz`
* Excludes unnecessary folders (`.git`, `.venv`, caches, `.env`)

---

### ☁️ Linode Setup

* Creates or updates firewall
* Opens only required ports:

| Port  | Purpose       |
| ----- | ------------- |
| 22    | SSH           |
| 6000  | A2A Adapter   |
| 8787  | Chat Backend  |
| 16000 | NANDA Wrapper |

* Generates SSH keypair
* Creates Linode instance
* Uploads project via SCP

---

### 🖥️ Remote Server Setup

On the server:

* Installs Python, Git, Supervisor
* Creates `ubuntu` user
* Extracts project to:

```
/home/ubuntu/mbta-agent
```

* Sets up Python virtual environment
* Installs dependencies
* Installs NANDA NEST
* Configures Supervisor programs:

| Service          | Port  |
| ---------------- | ----- |
| Alerts Agent     | 8781  |
| Planner Agent    | 8782  |
| StopFinder Agent | 8783  |
| Chat Backend     | 8787  |
| A2A Adapter      | 6000  |
| NANDA Wrapper    | 16000 |

---

### 🎉 Final Output Includes:

* Linode Instance ID
* Public IP
* Root password
* Generated NANDA Agent ID
* URLs:

```
Chat UI:           http://<IP>:8787
Chat API docs:     http://<IP>:8787/docs
A2A Adapter:       http://<IP>:6000/a2a
NANDA Wrapper:     http://<IP>:16000/a2a
```

---

# 🧪 6. Verify Deployment

### Test Chat Backend

```bash
curl http://<PUBLIC_IP>:8787/chat \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"routes"}]}'
```

---

### Test A2A Adapter

```bash
curl http://<PUBLIC_IP>:16000/a2a \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"content":{"text":"test","type":"text"},"role":"user","conversation_id":"t1"}'
```

A valid JSON response means the agent is functioning correctly.

---

# 🛡️ 7. Security Notes

* Firewall restricts all ports except required ones
* Root password can be auto-generated securely
* SSH keypair is isolated per deployment
* Supervisor ensures auto-restart on failure

---

# 🧹 8. Cleanup

Delete instance:

```bash
linode-cli linodes delete <INSTANCE_ID>
```

Delete firewall:

```bash
linode-cli firewalls delete <FIREWALL_ID>
```

---

