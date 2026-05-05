"use client";
import { useState } from "react";

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleRewrite = async () => {
    if (!text.trim()) {
      alert("Please enter some text");
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://localhost:8000/rewrite", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
      });

      const data = await res.json();
      setResult(data);
    } catch (error) {
      alert("Error connecting to backend");
      console.log(error);
    }

    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>PlagiarismAI</h1>

      {/* INPUT */}
      <textarea
        style={styles.textarea}
        placeholder="Paste your text here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      {/* BUTTON */}
      <button style={styles.button} onClick={handleRewrite}>
        Rewrite
      </button>

      {/* LOADING */}
      {loading && <p style={styles.loading}>Processing...</p>}

      {/* OUTPUT */}
      {result && (
        <>
          <div style={styles.outputContainer}>
            <div style={styles.box}>
              <h3>Original</h3>
              <p>{result.original}</p>
            </div>

            <div style={styles.box}>
              <h3>Rewritten</h3>
              <p>{result.rewritten}</p>
            </div>
          </div>

          {/* METRICS */}
          <div style={styles.metrics}>
            <p>Similarity: {(result.similarity * 100).toFixed(2)}%</p>
            <p>Plagiarism: {result.plagiarism_percent.toFixed(2)}%</p>
          </div>
        </>
      )}
    </div>
  );
}

const styles = {
  container: {
    maxWidth: "900px",
    margin: "auto",
    padding: "20px",
    fontFamily: "Arial",
    backgroundColor: "#0f172a",
    color: "white",
    minHeight: "100vh"
  },

  title: {
    textAlign: "center"
  },

  textarea: {
    width: "100%",
    height: "150px",
    padding: "10px",
    borderRadius: "8px",
    border: "none",
    marginTop: "10px"
  },

  button: {
    marginTop: "10px",
    padding: "10px 20px",
    backgroundColor: "#2563eb",
    color: "white",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer"
  },

  loading: {
    marginTop: "10px"
  },

  outputContainer: {
    display: "flex",
    gap: "20px",
    marginTop: "20px"
  },

  box: {
    flex: 1,
    background: "#1e293b",
    padding: "15px",
    borderRadius: "8px"
  },

  metrics: {
    marginTop: "20px",
    fontWeight: "bold"
  }
};