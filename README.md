
# Plagiarism AI Detection System

## 📌 Project Overview

The **Plagiarism AI Detection System** is a machine learning–based application designed to detect similarities between text documents and identify potential plagiarism. The system analyzes input text and compares it with reference documents or datasets to calculate similarity scores using Natural Language Processing (NLP) techniques.

This project helps educators, students, researchers, and content creators verify originality and maintain academic integrity.

---

## 🚀 Features

* Upload or paste text for plagiarism detection
* Compare text against stored datasets/documents
* Similarity score generation
* Highlight matched content sections
* User-friendly interface
* Fast processing using optimized NLP techniques
* Scalable architecture for future dataset expansion

---

## 🛠️ Technologies Used

**Frontend:**

* HTML
* CSS
* JavaScript

**Backend:**

* Python (Flask)

**Libraries & Tools:**

* scikit-learn
* NumPy
* Pandas
* NLTK / spaCy
* TF-IDF Vectorizer
* Cosine Similarity

---

## 📂 Project Structure

```
plagiarism-ai-detection/
│
├── static/                # CSS, JS, images
├── templates/             # HTML files
├── dataset/               # Reference documents
├── model/                 # Trained model files
├── app.py                 # Main Flask application
├── utils.py               # Helper functions
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally:

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/plagiarism-ai-detection.git
cd plagiarism-ai-detection
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows:**

```bash
.venv\\Scripts\\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python app.py
```

Now open your browser and go to:

```
http://127.0.0.1:5000
```

---

## 🧠 How It Works

1. User uploads or enters text.
2. Text preprocessing is applied:

   * Lowercasing
   * Stopword removal
   * Tokenization
   * Lemmatization
3. TF-IDF converts text into numerical vectors.
4. Cosine similarity compares input text with dataset documents.
5. Similarity percentage is generated.
6. Matching sections are highlighted in results.

---

## 📊 Example Output

| Input Text       | Matched Document | Similarity Score |
| ---------------- | ---------------- | ---------------- |
| Sample paragraph | doc3.txt         | 82%              |

---

## 🔮 Future Improvements

* Integration with large academic databases
* Real-time web plagiarism detection
* Deep learning–based semantic similarity detection
* Multi-language plagiarism detection
* User authentication system
* Cloud deployment support

---

## 🤝 Contributing

Contributions are welcome!

Steps to contribute:

1. Fork the repository
2. Create a new branch
3. Make changes
4. Commit updates
5. Submit a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Joita Paul**
**Manisha Debnath**
B.Tech CSE Student | AI & Full‑Stack Enthusiast

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!
