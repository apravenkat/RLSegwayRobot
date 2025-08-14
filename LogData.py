import serial
import time
import csv

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(1)  # Let Arduino reset

# Keep everything inside the `with` block
with open('test_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)

    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8').strip()
                writer.writerow([line])
                file.flush()
    except KeyboardInterrupt:
        print("Logging stopped.")
        ser.close()