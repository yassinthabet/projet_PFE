import json
import csv
import paho.mqtt.client as mqtt

broker_address = "127.0.0.1"
broker_port = 1883
topic = "vehicle_data"


def save_to_csv(data):
    with open('data.csv', 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        
        writer.writerow(data)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(topic, qos=0)
    else:
        print("Failed to connect to broker with code", rc)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())  
        print(f"Received message on topic '{msg.topic}': {payload}")
        save_to_csv(payload)  
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {msg.payload.decode()}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker_address, broker_port, 60)

client.loop_forever()
