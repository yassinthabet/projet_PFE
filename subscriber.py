import paho.mqtt.client as mqtt

broker_address = "127.0.0.1"
broker_port = 1883
topic = "vehicle_data"
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(topic)
    else:
        print("Failed to connect to broker with code", rc)

def on_message(client, userdata, msg):
    print(f"Received message on topic '{msg.topic}': {msg.payload.decode()}")

client = mqtt.Client()


client.on_connect = on_connect
client.on_message = on_message


client.connect(broker_address, broker_port, 60)

client.loop_forever()