# SSH Tunnel Setup

This directory holds the SSH private key for the reverse tunnel to your cloud server.

## Quick Start

### 1. Generate SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -f tunnel/id_rsa -N "" -C "actus-tunnel"
```

### 2. Copy public key to cloud server

```bash
ssh-copy-id -i tunnel/id_rsa.pub root@YOUR_CLOUD_SERVER_IP
```

### 3. Enable GatewayPorts on cloud server

SSH into your cloud server and run:

```bash
sudo sed -i '/^#\?GatewayPorts/c\GatewayPorts yes' /etc/ssh/sshd_config
sudo systemctl restart sshd
```

### 4. Configure .env

```bash
TUNNEL_REMOTE_HOST=YOUR_CLOUD_SERVER_IP
TUNNEL_REMOTE_PORT=18082
TUNNEL_SSH_USER=root
TUNNEL_SSH_PORT=22
TUNNEL_SSH_KEY_PATH=./tunnel/id_rsa
```

### 5. Start

```bash
docker compose up -d tunnel
```

### 6. Verify

From your phone or any external network:

```bash
curl http://YOUR_CLOUD_SERVER_IP:18082/docs
```

Should return the Actus API docs page.

## How it works

```
Phone ──HTTP──> Cloud:18082 ──SSH tunnel──> Docker(api):8000
```

The `tunnel` container runs `autossh` which:
- Maintains a persistent SSH reverse tunnel
- Auto-reconnects on network failures
- Sends keepalive every 15 seconds
- Forwards cloud server's port 18082 to the `api` container's port 8000

## Security

- The SSH key in this directory should NOT be committed to git
- Add `tunnel/id_rsa` to `.gitignore`
