import serial
import serial.tools.list_ports
import pandas as pd
import openpyxl
import os

# List available COM ports
ports = serial.tools.list_ports.comports()
portList = []

print("Available COM ports:")
for onePort in ports:
    portList.append(onePort.device)
    print(onePort.device)

# Get user input for port selection
val = input("Select port (e.g., COM4): ").strip()

if val not in portList:
    print("Invalid port selection!")
    exit()

try:
    serialInst = serial.Serial(port=val, baudrate=9600, timeout=1)
    print(f"Connected to {val}. Reading data...\n")
except serial.SerialException as e:
    print(f"Error: {e}")
    exit()

excel_file = "sensor_data.xlsx"

if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=["Temperature (°C)", "SpO₂ (%)", "Heart Rate (BPM)"])
    df.to_excel(excel_file, index=False)
 
# Initialize variables
temperature = None
blood_oxygen = None
heart_rate = None

while True:
    try:
        if serialInst.in_waiting > 0:
            data = serialInst.readline().decode('utf-8').strip()
            print(f"Received: {data}")

            if "Temperature" in data:
                temperature = float(data.split("=")[1].strip())
            elif "SpO2" in data:
                blood_oxygen = float(data.split("=")[1].strip())
            elif "Heart Rate" in data:
                heart_rate = float(data.split("=")[1].strip())

            # If all data is collected, save to Excel
            if temperature is not None and blood_oxygen is not None and heart_rate is not None:
                df = pd.read_excel(excel_file)
                new_data = pd.DataFrame([[temperature, blood_oxygen, heart_rate]], 
                                        columns=["Temperature (°C)", "SpO₂ (%)", "Heart Rate (BPM)"])
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_excel(excel_file, index=False)
                print(f"Data saved to {excel_file}")

                # Reset after saving
                temperature = None
                blood_oxygen = None
                heart_rate = None

    except Exception as e:
        print(f"Error: {e}")
