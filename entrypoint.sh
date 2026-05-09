#!/usr/bin/env bash
# RunPod entrypoint:
#   - sets up SSH (RunPod's web SSH expects sshd)
#   - chooses one of: train | wait | bash
set -euo pipefail

# 1. SSH for RunPod web SSH access. Authorized keys come from RunPod's
#    PUBLIC_KEY env var if set (their default), or from a mounted file.
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
# Allow root login + restart sshd.
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
service ssh start || /usr/sbin/sshd

# 2. Pre-create workspace dirs.
mkdir -p /workspace "${DATA_DIR}" "${OUTPUT_DIR}"

cmd="${1:-wait}"
case "$cmd" in
  train)
    exec python /opt/run_finetune.py
    ;;
  bash|sh)
    exec bash
    ;;
  wait)
    echo "[entrypoint] Pod ready."
    echo "[entrypoint] To get data in:  runpodctl receive <code>   (run 'runpodctl send <file>' locally)"
    echo "[entrypoint] To start training:  /opt/entrypoint.sh train"
    echo "[entrypoint] Or set CMD=train when you start the pod."
    # tail -f keeps PID 1 alive; container stays up so user can SSH in.
    tail -f /dev/null
    ;;
  *)
    exec "$@"
    ;;
esac
