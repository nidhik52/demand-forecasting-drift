producer.py
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

events = [
    {"type": "DRIFT", "sku": "PETS-003"},
    {"type": "ORDER", "sku": "AUTO-002", "qty": 120}
]

for event in events:
    producer.send("demand-events", event)
    print("Sent:", event)

producer.flush()