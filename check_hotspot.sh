# TO RUN ON PI 5

#!/bin/bash

echo "🔧 Checking Raspberry Pi Hotspot Health..."

# Check wlan0 status
echo -n "📡 wlan0 status: "
if ip link show wlan0 | grep -q "state UP"; then
    echo "✅ UP"
else
    echo "❌ DOWN"
    echo "   ➤ Bringing up wlan0..."
    sudo ip link set wlan0 up
fi

# Check wlan0 IP address
WLAN_IP=$(ip -4 addr show wlan0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
if [[ "$WLAN_IP" == "192.168.4.1" ]]; then
    echo "🌐 wlan0 IP: ✅ $WLAN_IP"
else
    echo "🌐 wlan0 IP: ❌ $WLAN_IP"
    echo "   ➤ Forcing static IP..."
    sudo ip addr flush dev wlan0
    sudo ip addr add 192.168.4.1/24 dev wlan0
fi

# Restart dnsmasq
echo -n "🧰 Restarting dnsmasq..."
sudo systemctl restart dnsmasq && echo " ✅ OK" || echo " ❌ Failed"

# Restart hostapd
echo -n "📶 Restarting hostapd..."
sudo systemctl restart hostapd && echo " ✅ OK" || echo " ❌ Failed"

# Check service status
echo ""
echo "🧪 Final Status Check:"
systemctl is-active hostapd | grep -q active && echo "✅ hostapd running" || echo "❌ hostapd not running"
systemctl is-active dnsmasq | grep -q active && echo "✅ dnsmasq running" || echo "❌ dnsmasq not running"

echo ""
echo "✅ Script complete. Your Pi should now be broadcasting the hotspot."
echo "👉 Reconnect from your PC and run: ipconfig + ping 192.168.4.1"
