from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load tokenizer and model for summarization
tokenizer = AutoTokenizer.from_pretrained("Yihui/t5-small-text-summary-generation")
model = AutoModelForSeq2SeqLM.from_pretrained("Yihui/t5-small-text-summary-generation")

# Function to summarize text
def summarize_text(text, max_length=50, min_length=25, num_beams=2):
    # Tokenize the input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=num_beams,
        early_stopping=True
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission and summarize text
@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text from the form
    input_text = request.form['input_text']

    # Summarize the text
    summary = summarize_text(input_text)

    # Render the result on a new page
    return render_template('index.html', original_text=input_text, summary_text=summary)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)