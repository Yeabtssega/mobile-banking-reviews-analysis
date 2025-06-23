# explain_amharic_ner.py

from transformers import pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np

# Load your trained NER model and tokenizer
# Replace "amharic-ner-xlmroberta" with your saved model folder or Huggingface repo name
model_name_or_path = "amharic-ner-xlmroberta"  

print("Loading model and tokenizer...")
ner_pipeline = pipeline(
    "ner",
    model=model_name_or_path,
    aggregation_strategy="simple"  # aggregates tokens into entities
)

# Define entity label to index mapping (adjust based on your labels)
entity_to_index = {
    "O": 0,
    "LOC": 1,
    "PRICE": 2,
    "PRODUCT": 3
}

def predict_proba(texts):
    outputs = []
    for text in texts:
        preds = ner_pipeline(text)
        # Initialize probabilities array for each entity class
        output = [0]*len(entity_to_index)
        for pred in preds:
            label = pred['entity_group']
            idx = entity_to_index.get(label, 0)  # default to "O"
            # Save max score if multiple predictions per class
            output[idx] = max(output[idx], pred['score'])
        outputs.append(output)
    return np.array(outputs)

# Create the LIME explainer
explainer = LimeTextExplainer(class_names=list(entity_to_index.keys()))

# Text to explain
text_to_explain = "ዋጋ 1000 ብር በቦሌ መድሐኔዓለም"

print(f"Explaining prediction for: {text_to_explain}")

# Generate explanation
exp = explainer.explain_instance(text_to_explain, predict_proba, num_features=10)

# Print the explanation
print("\nLIME explanation (feature weights):")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight}")

# Optional: Save explanation visualization as HTML
exp.save_to_file("lime_explanation.html")
print("\nExplanation saved as lime_explanation.html")
