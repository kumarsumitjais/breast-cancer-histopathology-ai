import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

type Prediction = {
  filename: string;
  predicted_class: "benign" | "malignant";
  malignant_probability: number;
  threshold: number;
  model_id: string;
};

type Item = {
  id: number;
  file: File;
  preview: string;
  aspect?: number; // original width / height, used to un-squash the 224x224 heatmap
  status: "loading" | "done" | "error";
  result?: Prediction;
  error?: string;
  heatmap?: string;
  heatmapLoading?: boolean;
  heatmapError?: string;
};

let nextId = 0;

async function errorText(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`;
  } catch {
    return `Request failed (${res.status})`;
  }
}

async function post(path: string, file: File): Promise<Response> {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${API}${path}`, { method: "POST", body: form });
}

const OFFLINE = "Cannot reach the API. Is the backend running?";

export default function App() {
  const [health, setHealth] = useState<"checking" | "ready" | "no_model" | "offline">("checking");
  const [items, setItems] = useState<Item[]>([]);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then((h) => setHealth(h.model_loaded ? "ready" : "no_model"))
      .catch(() => setHealth("offline"));
  }, []);

  const update = (id: number, patch: Partial<Item>) =>
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...patch } : it)));

  async function addFiles(files: FileList | null) {
    if (!files?.length) return;
    const added: Item[] = Array.from(files).map((file) => ({
      id: nextId++,
      file,
      preview: URL.createObjectURL(file),
      status: "loading",
    }));
    setItems((prev) => [...added, ...prev]);
    // ponytail: sequential requests (model runs on CPU); parallelise if the backend scales out.
    for (const it of added) {
      try {
        const res = await post("/predict", it.file);
        if (res.ok) update(it.id, { status: "done", result: await res.json() });
        else update(it.id, { status: "error", error: await errorText(res) });
      } catch {
        update(it.id, { status: "error", error: OFFLINE });
      }
    }
  }

  async function explain(it: Item) {
    update(it.id, { heatmapLoading: true, heatmapError: undefined });
    try {
      const res = await post("/explain", it.file);
      if (res.ok) update(it.id, { heatmapLoading: false, heatmap: URL.createObjectURL(await res.blob()) });
      else update(it.id, { heatmapLoading: false, heatmapError: await errorText(res) });
    } catch {
      update(it.id, { heatmapLoading: false, heatmapError: OFFLINE });
    }
  }

  function clearAll() {
    items.forEach((it) => {
      URL.revokeObjectURL(it.preview);
      if (it.heatmap) URL.revokeObjectURL(it.heatmap);
    });
    setItems([]);
  }

  const healthLabel = {
    checking: "Checking API…",
    ready: "Model ready",
    no_model: "Model weights missing",
    offline: "API offline",
  }[health];

  return (
    <>
      <header>
        <h1>Breast Histopathology Classifier</h1>
        <span className={`badge ${health}`}>{healthLabel}</span>
      </header>

      <main>
        <p className="warning">
          <strong>Research use only. Not a medical device.</strong> Do not use these results to diagnose, screen for,
          or make treatment decisions about breast cancer.
        </p>

        <label
          className={`dropzone ${dragging ? "dragging" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            addFiles(e.dataTransfer.files);
          }}
        >
          <input
            type="file"
            multiple
            accept="image/jpeg,image/png,image/tiff"
            onChange={(e) => {
              addFiles(e.target.files);
              e.target.value = ""; // allow re-selecting the same file
            }}
          />
          <strong>Drop images here or click to choose</strong>
          <span>
            JPEG, PNG or TIFF · up to 12 MB · sides 224–2048 px · aspect ratio up to 2:1 (e.g. 700×460)
          </span>
        </label>

        {items.length > 0 && (
          <div className="toolbar">
            <span>{items.length} image(s)</span>
            <button onClick={clearAll}>Clear all</button>
          </div>
        )}

        <div className="results">
          {items.map((it) => (
            <article key={it.id} className="card">
              <div className="images">
                <figure>
                  <img
                    src={it.preview}
                    alt={`Uploaded ${it.file.name}`}
                    onLoad={(e) => update(it.id, { aspect: e.currentTarget.naturalWidth / e.currentTarget.naturalHeight })}
                    onError={(e) => (e.currentTarget.style.display = "none")}
                  />
                  <figcaption>Original</figcaption>
                </figure>
                {it.heatmap && (
                  <figure>
                    <img src={it.heatmap} alt={`Grad-CAM heatmap for ${it.file.name}`} style={{ aspectRatio: it.aspect }} />
                    <figcaption>Grad-CAM</figcaption>
                  </figure>
                )}
              </div>

              <div className="filename" title={it.file.name}>{it.file.name}</div>

              {it.status === "loading" && <p className="muted">Analysing…</p>}
              {it.status === "error" && <p className="error">{it.error}</p>}
              {it.result && <ResultView result={it.result} />}

              {it.result && !it.heatmap && (
                <button onClick={() => explain(it)} disabled={it.heatmapLoading}>
                  {it.heatmapLoading ? "Generating heatmap…" : "Show Grad-CAM heatmap"}
                </button>
              )}
              {it.heatmapError && <p className="error">{it.heatmapError}</p>}
              {it.heatmap && <HeatmapExplanation />}
            </article>
          ))}
        </div>
      </main>

      <footer>
        Images that differ from the BreakHis training data (stain, scanner, magnification) may give unreliable
        results. Uploaded images are not stored.
      </footer>
    </>
  );
}

function HeatmapExplanation() {
  return (
    <details className="explanation">
      <summary>How to read this heatmap</summary>
      <p>
        This heatmap was made with a technique called Grad-CAM. It shows which parts of the image had the most
        influence on the model's malignant probability. The colours are laid over a copy of your image: red areas
        had the strongest influence on the score, blue areas had little or none, and the colours in between had a
        moderate effect.
      </p>
      <p>
        The heatmap is calculated from the model's last layer of image features, which divides the image into a
        coarse 7×7 grid. It is then smoothed and stretched to fit the image, so it highlights broad regions, not
        individual cells or nuclei.
      </p>
      <p>
        The heatmap always describes the malignant score. When the result is malignant, red areas are where the
        model found the most evidence for that result. When the result is benign, red areas are the parts that
        still pushed the score towards malignant, even though the total stayed below the threshold.
      </p>
      <p>
        A red area shows where the model looked, not that the area contains cancer. The model can focus on things
        unrelated to disease, such as staining differences, tissue edges, blur or empty background, and the heatmap
        highlights those just as brightly. It is a tool for understanding how the model behaves. It has not been
        checked against pathologists' annotations and must never be used to locate or rule out disease.
      </p>
    </details>
  );
}

function ResultView({ result }: { result: Prediction }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  return (
    <div className="result">
      <span className={`label ${result.predicted_class}`}>{result.predicted_class}</span>
      <div className="bar" role="img" aria-label={`Malignant probability ${pct(result.malignant_probability)}`}>
        <div className="fill" style={{ width: pct(result.malignant_probability) }} />
        <div className="threshold" style={{ left: pct(result.threshold) }} />
      </div>
      <p className="small">
        Malignant probability <strong>{pct(result.malignant_probability)}</strong> · classified malignant when ≥{" "}
        {pct(result.threshold)}
      </p>
    </div>
  );
}
