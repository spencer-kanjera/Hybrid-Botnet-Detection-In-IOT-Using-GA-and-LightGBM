from scapy.all import IP, TCP, UDP, ICMP, send, sr1, RandShort
import random
import time
import threading
import signal

# Shared variable to control simulation shutdown
stop_simulation = threading.Event()  # Use threading.Event for thread synchronization

# Target IP address (always set to `.158`)
target_ip = "172.16.15.158"

# Function to handle Ctrl+C (SIGINT)
def signal_handler(sig, frame):
    global stop_simulation
    print("\n[INFO] Ctrl+C pressed! Stopping traffic simulation...")
    stop_simulation.set()  # Signal all threads to stop

# Function to check if the target IP is reachable
def is_target_reachable():
    print(f"[INFO] Checking if {target_ip} is reachable...")
    icmp_packet = IP(dst=target_ip) / ICMP()  # ICMP packet (ping)
    response = sr1(icmp_packet, timeout=2, verbose=False)
    if response:
        print(f"[INFO] Target {target_ip} is reachable.")
        return True
    else:
        print(f"[ERROR] Target {target_ip} is not reachable. Exiting...")
        return False

# Function to generate normal traffic
def generate_normal_traffic():
    print("[INFO] Generating normal traffic...")
    while not stop_simulation.is_set():
        http_packet = IP(dst=target_ip) / TCP(sport=RandShort(), dport=80, flags="S")
        send(http_packet, verbose=False)
        dns_packet = IP(dst=target_ip) / UDP(sport=RandShort(), dport=53) / ("X" * 128)
        send(dns_packet, verbose=False)
        time.sleep(random.uniform(0.05, 0.2))  # Randomized delay

# Function to generate DoS traffic
def generate_dos_traffic():
    print("[INFO] Generating DoS traffic...")
    while not stop_simulation.is_set():
        dos_packet = IP(dst=target_ip) / TCP(sport=RandShort(), dport=80, flags="S") / ("X" * 1024)
        send(dos_packet, verbose=False)

# Function to generate brute force traffic
def generate_brute_force_traffic():
    print("[INFO] Generating brute force traffic...")
    while not stop_simulation.is_set():
        packet = IP(dst=target_ip) / TCP(sport=RandShort(), dport=22, flags="S")
        send(packet, verbose=False)
        time.sleep(0.1)

# Function to generate port scanning traffic
def generate_port_scan_traffic():
    print("[INFO] Generating port scan traffic...")
    while not stop_simulation.is_set():
        port = random.randint(1, 65535)
        scan_packet = IP(dst=target_ip) / TCP(sport=RandShort(), dport=port, flags="S")
        send(scan_packet, verbose=False)

# Function to generate botnet-style traffic
def generate_botnet_traffic():
    print("[INFO] Generating botnet-style traffic...")
    while not stop_simulation.is_set():
        bot_packet = IP(dst=target_ip) / UDP(sport=RandShort(), dport=random.randint(1024, 65535)) / ("X" * 512)
        send(bot_packet, verbose=False)
        time.sleep(0.05)

# Function to start all traffic simulations in parallel
def start_simulation():
    global stop_simulation

    # Check if the target IP is reachable
    if not is_target_reachable():
        return  # Exit if the target is unreachable

    # Define threads for different traffic types
    threads = [
        threading.Thread(target=generate_normal_traffic),
        threading.Thread(target=generate_dos_traffic),
        threading.Thread(target=generate_brute_force_traffic),
        threading.Thread(target=generate_port_scan_traffic),
        threading.Thread(target=generate_botnet_traffic),
    ]

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Start all threads
    for thread in threads:
        thread.start()

    print("[INFO] Traffic simulation started. Press Ctrl+C to stop.")

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("[INFO] All traffic threads have stopped.")

if __name__ == "__main__":
    start_simulation()