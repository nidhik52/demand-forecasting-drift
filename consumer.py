from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "demand-events",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True
)

print("Listening to events...\n")

for msg in consumer:
    print("Received:", msg.value)