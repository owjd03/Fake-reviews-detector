from src.predict import predict

if __name__ == "__main__":
    while True:
        text = input("Enter review text: ")
        if text.lower() == "exit":
            break
        print("Prediction:", predict(text))
    print("thank you")
