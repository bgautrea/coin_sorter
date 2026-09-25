#!/usr/bin/env bash
# Control the coin-webcal user service (deploy/coin-webcal.service).
#
#   scripts/webcal.sh install   copy the unit, enable it at boot (needs sudo once for linger)
#   scripts/webcal.sh start|stop|restart|status
#   scripts/webcal.sh logs      follow the journal
#   scripts/webcal.sh enable|disable   toggle start-at-boot
set -euo pipefail

unit=coin-webcal
repo="$(cd "$(dirname "$0")/.." && pwd)"
sc() { systemctl --user "$@"; }

case "${1:-status}" in
    install)
        mkdir -p ~/.config/systemd/user
        cp "$repo/deploy/$unit.service" ~/.config/systemd/user/
        sc daemon-reload
        sc enable "$unit"
        if [[ "$(loginctl show-user "$USER" -p Linger --value)" != yes ]]; then
            echo "Enabling linger so the service starts at boot without a login (sudo):"
            sudo loginctl enable-linger "$USER"
        fi
        echo "Installed and enabled. Start it now with: $0 start"
        ;;
    start|stop|restart|status|enable|disable)
        sc "$1" "$unit"
        ;;
    logs)
        journalctl --user -u "$unit" -f
        ;;
    *)
        echo "usage: $0 {install|start|stop|restart|status|logs|enable|disable}" >&2
        exit 2
        ;;
esac
