from scapy.all import sniff, IP, TCP, UDP
import numpy as np
import pandas as pd
import joblib
import logging

# Configure logging for anomalies
logging.basicConfig(filename="anomalies.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load your pre-trained LightGBM model
model = joblib.load("models_and_data\\final_model.joblib")  # Replace with your model file path

# Replace this with the IP address you want to filter
target_ip = "172.16.14.126"

# Function to extract features from a packet (matches the specified feature set)
def extract_features(packet):
    features = {
        "Destination Port": packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else None),
        "Flow Duration": packet.time if hasattr(packet, "time") else None,
        "Fwd Packet Length Min": len(packet) if TCP in packet and packet[TCP].flags == "F" else None,
        "Bwd Packet Length Max": len(packet) if TCP in packet and packet[TCP].flags == "R" else None,
        "Bwd Packet Length Min": len(packet) if TCP in packet and packet[TCP].flags == "A" else None,
        "Bwd Packet Length Mean": len(packet) if hasattr(packet, "len") else None,
        "Bwd Packet Length Std": np.std([len(packet)]) if hasattr(packet, "len") else None,
        "Flow IAT Mean": packet.time if hasattr(packet, "time") else None,
        "Flow IAT Std": np.std([packet.time]) if hasattr(packet, "time") else None,
        "Flow IAT Max": max([packet.time]) if hasattr(packet, "time") else None,
        "Fwd IAT Total": sum([len(packet)]) if hasattr(packet, "len") else None,
        "Fwd IAT Mean": np.mean([len(packet)]) if hasattr(packet, "len") else None,
        "Fwd IAT Std": np.std([len(packet)]) if hasattr(packet, "len") else None,
        "Fwd IAT Max": max([len(packet)]) if hasattr(packet, "len") else None,
        "Min Packet Length": len(packet) if hasattr(packet, "len") else None,
        "Max Packet Length": len(packet) if hasattr(packet, "len") else None,
        "Packet Length Mean": np.mean([len(packet)]) if hasattr(packet, "len") else None,
        "Packet Length Std": np.std([len(packet)]) if hasattr(packet, "len") else None,
        "Packet Length Variance": np.var([len(packet)]) if hasattr(packet, "len") else None,
        "FIN Flag Count": 1 if TCP in packet and packet[TCP].flags == "F" else 0,
        "URG Flag Count": 1 if TCP in packet and packet[TCP].flags == "U" else 0,
        "Average Packet Size": np.mean([len(packet)]) if hasattr(packet, "len") else None,
        "Avg Bwd Segment Size": len(packet) if TCP in packet and packet[TCP].flags == "R" else None,
        "Idle Mean": packet.time if hasattr(packet, "time") else None,
        "Idle Max": max([packet.time]) if hasattr(packet, "time") else None,
        "Idle Min": min([packet.time]) if hasattr(packet, "time") else None,
    }
    return features

# Function to process and classify a packet
def classify_packet(packet):
    if packet.haslayer("IP") and (packet["IP"].src == target_ip or packet["IP"].dst == target_ip):
        # Extract features from the packet
        features = extract_features(packet)
        print("[INFO] Packet features:", features)  # Debugging output

        # Convert features into a DataFrame for the model
        features_df = pd.DataFrame([features])

        # Predict anomaly using the LightGBM model
        prediction = model.predict(features_df.fillna(0))[0]  # Replace NaN with 0 for missing values

        if prediction == 1:  # Anomalous traffic
            print("[ALERT] Anomalous traffic detected:", features)
            logging.info(f"Anomalous traffic: {features}")  # Save anomaly in the log file
        else:  # Normal traffic
            print("[INFO] Normal traffic detected:", features)

# Start sniffing on the Ethernet interface with filtering and classification
def start_sniffing(interface):
    print(f"[INFO] Capturing packets for IP address: {target_ip}")
    sniff(iface=interface, prn=classify_packet, store=False)  # Capture and process packets in real-time

if __name__ == "__main__":
    network_interface = "Ethernet"  # Replace with your correct Ethernet interface name
    start_sniffing(network_interface)