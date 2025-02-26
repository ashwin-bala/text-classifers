from transformers import pipeline
import csv
classifier = pipeline("text-classification", model="Falconsai/intent_classification")

def setUp():
    bank_content = []
    with open('test-content.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            bank_content.append(row[0])
    return bank_content

bank_content = setUp()

def classify_messages(classifier, messages):
    if not messages:
        print("No content found in messages. Exiting the program.")
        exit()

    # Example usage
    for message in messages:
        result = classifier(message)
        print("Message:", message, "Result:", result)


classify_messages(classifier, bank_content)