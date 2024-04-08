import json
import csv
import paho.mqtt.client as mqtt
import os 

broker_address = "192.168.7.2"
broker_port = 1883
topic = "vehicle_data"

csv_folder = "vehicle_data_csv"
csv_file = None

def create_csv_file(video_name):
    csv_file_path = os.path.join(csv_folder, f"{video_name}.csv")
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Color", "Make", "Model", "Registration", "Country"])
    return csv_file_path

def write_to_csv(data, csv_file):
    with open(csv_file.name, mode='a', newline='') as file:  
        writer = csv.writer(file)
        writer.writerow([data['class'], data['classificators'][0]['color'], data['classificators'][0]['make'],
                         data['classificators'][0]['model'], data['registration'], data['classificators'][0]['country']])

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(topic, qos=0)
    else:
        print("Failed to connect to broker with code", rc)

def on_message(client, userdata, msg):
    global csv_file

    print(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")
    try:
        data = json.loads(msg.payload.decode())
        if 'video_name' in data:
            video_name = data['video_name']
            if csv_file:
                csv_file.close() 
            csv_file = open(create_csv_file(video_name), mode='a', newline='')  
            print(f"Created CSV file for video: {csv_file.name}")
        else:
            write_to_csv(data, csv_file) 
            print("Data written to CSV successfully")
    except Exception as e:
        print("Error processing JSON data:", e)

client = mqtt.Client()

client.on_connect = on_connect
client.on_message = on_message

client.connect(broker_address, broker_port, 60)

client.loop_forever()