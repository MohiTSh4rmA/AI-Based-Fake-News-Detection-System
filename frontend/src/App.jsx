import { useState } from "react";

function App() {
  const [news, setNews] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeNews = async () => {
    if (!news.trim()) return;
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:9000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ news: news }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Backend not connected 🚨");
    }

    setLoading(false);
  };

  return (
    <div style={{ backgroundColor: "#0f172a", minHeight: "100vh", color: "white", padding: "40px", textAlign: "center" }}>
      <h1 style={{ fontSize: "36px", marginBottom: "30px" }}>
        🧠 AI Fake News Detection System
      </h1>

      <textarea
        style={{ width: "60%", padding: "15px", borderRadius: "10px" }}
        rows="6"
        placeholder="Paste news article or headline here..."
        value={news}
        onChange={(e) => setNews(e.target.value)}
      />

      <br />

      <button
        onClick={analyzeNews}
        style={{
          marginTop: "20px",
          padding: "10px 20px",
          backgroundColor: "#2563eb",
          border: "none",
          borderRadius: "8px",
          color: "white",
          cursor: "pointer"
        }}
      >
        {loading ? "Analyzing..." : "Analyze News"}
      </button>

      {result && (
        <div style={{ marginTop: "40px", backgroundColor: "#1e293b", padding: "20px", borderRadius: "10px" }}>
          <h2>{result.prediction}</h2>
          <p>Confidence: {result.confidence_percentage}%</p>
        </div>
      )}
    </div>
  );
}

export default App;