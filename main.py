import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from predict import predict_message

print("📧 Spam Mail Detector Running...\n")

while True:
    msg = input("Enter message (or type 'exit'): ")

    if msg.lower() == "exit":
        break

    result = predict_message(msg)
    print("Prediction:", result)
