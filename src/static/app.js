/**
 * Pinecone Document Q&A & Hybrid RAG Showcase
 * Client Application Logic
 */

let currentSessionId = null;

// Initialize on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupRangeSliders();
  setupQuickPills();
  setupSampleLoader();
  setupForms();
  fetchTelemetry();
  fetchCatalog();
});

// Tab Switching
function setupTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
      }

      if (targetId === "tab-telemetry") {
        fetchTelemetry();
      } else if (targetId === "tab-studio") {
        fetchCatalog();
      }
    });
  });
}

// Range Sliders Value Updates
function setupRangeSliders() {
  const chunkSizeInput = document.getElementById("doc-chunk-size");
  const chunkOverlapInput = document.getElementById("doc-chunk-overlap");
  const topKInput = document.getElementById("search-top-k");
  const alphaInput = document.getElementById("search-alpha");

  if (chunkSizeInput) {
    chunkSizeInput.addEventListener("input", (e) => {
      document.getElementById("val-chunk-size").textContent = e.target.value;
    });
  }

  if (chunkOverlapInput) {
    chunkOverlapInput.addEventListener("input", (e) => {
      document.getElementById("val-chunk-overlap").textContent = e.target.value;
    });
  }

  if (topKInput) {
    topKInput.addEventListener("input", (e) => {
      document.getElementById("val-top-k").textContent = e.target.value;
    });
  }

  if (alphaInput) {
    alphaInput.addEventListener("input", (e) => {
      const val = parseFloat(e.target.value);
      document.getElementById("val-alpha").textContent = val.toFixed(2);
      const label = document.getElementById("alpha-label");
      if (val === 1.0) {
        label.textContent = "Pure Dense Semantic (100% Dense)";
      } else if (val === 0.0) {
        label.textContent = "Pure BM25 Sparse Keyword (100% Sparse)";
      } else {
        const densePct = Math.round(val * 100);
        const sparsePct = 100 - densePct;
        label.textContent = `Balanced Hybrid (${densePct}% Dense / ${sparsePct}% Sparse)`;
      }
    });
  }
}

// Quick Pill Fillers
function setupQuickPills() {
  document.querySelectorAll(".pill-btn").forEach((pill) => {
    pill.addEventListener("click", () => {
      const ns = pill.getAttribute("data-set-ns");
      const nsInput = document.getElementById("doc-namespace");
      if (nsInput && ns) {
        nsInput.value = ns;
      }
    });
  });
}

// Sample Document Loader
function setupSampleLoader() {
  const btn = document.getElementById("btn-load-sample");
  if (!btn) return;

  btn.addEventListener("click", () => {
    document.getElementById("doc-title").value = "Pinecone Vector Database Architecture";
    document.getElementById("doc-category").value = "Databases";
    document.getElementById("doc-author").value = "Architecture Team";
    document.getElementById("doc-namespace").value = "knowledge-base";
    document.getElementById("doc-content").value =
      "Pinecone is a managed, cloud-native vector database designed for high-scale machine learning applications. " +
      "It supports live upserts and sub-millisecond nearest-neighbor search across billions of high-dimensional dense vectors. " +
      "Pinecone partitions vector indexes into isolated namespaces, enabling secure multi-tenant architectures without data leakage. " +
      "Hybrid search in Pinecone combines dense float vectors with sparse BM25 token frequencies, balancing semantic understanding with exact keyword precision. " +
      "Metadata filtering allows real-time evaluation of boolean and comparison operators during vector similarity search.";
  });
}

// Form Handlers
function setupForms() {
  // 1. Ingestion Form
  const formIngest = document.getElementById("form-ingest");
  if (formIngest) {
    formIngest.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn-submit-ingest");
      btn.disabled = true;
      btn.textContent = "Ingesting & Embedding...";

      const payload = {
        title: document.getElementById("doc-title").value.trim(),
        content: document.getElementById("doc-content").value.trim(),
        category: document.getElementById("doc-category").value.trim(),
        author: document.getElementById("doc-author").value.trim(),
        namespace: document.getElementById("doc-namespace").value.trim(),
        chunk_size: parseInt(document.getElementById("doc-chunk-size").value, 10),
        chunk_overlap: parseInt(document.getElementById("doc-chunk-overlap").value, 10),
      };

      try {
        const res = await fetch("/api/documents", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const err = await res.json();
          alert("Error ingesting document: " + (err.detail || res.statusText));
          return;
        }

        const data = await res.json();
        alert(`Document '${data.title}' successfully ingested! Created ${data.chunk_count} vector chunks.`);
        document.getElementById("doc-content").value = "";
        fetchCatalog();
        fetchTelemetry();
      } catch (err) {
        alert("Failed to connect to API: " + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "Ingest & Embed to Pinecone";
      }
    });
  }

  // 2. Hybrid Search Form
  const formSearch = document.getElementById("form-search");
  if (formSearch) {
    formSearch.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn-run-search");
      btn.disabled = true;
      btn.textContent = "Searching...";

      const payload = {
        query_text: document.getElementById("search-query").value.trim(),
        namespace: document.getElementById("search-namespace").value,
        alpha: parseFloat(document.getElementById("search-alpha").value),
        top_k: parseInt(document.getElementById("search-top-k").value, 10),
      };

      try {
        const res = await fetch("/api/search/hybrid", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const err = await res.json();
          alert("Search error: " + (err.detail || res.statusText));
          return;
        }

        const data = await res.json();
        renderSearchResults(data.matches);
      } catch (err) {
        alert("Search failed: " + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "Execute Hybrid Search";
      }
    });
  }

  // 3. Conversational Chat Form
  const formChat = document.getElementById("form-chat");
  if (formChat) {
    formChat.addEventListener("submit", async (e) => {
      e.preventDefault();
      const input = document.getElementById("chat-input-text");
      const userText = input.value.trim();
      if (!userText) return;

      appendUserBubble(userText);
      input.value = "";
      const sendBtn = document.getElementById("btn-send-chat");
      sendBtn.disabled = true;

      const payload = {
        session_id: currentSessionId,
        message: userText,
        namespace: "knowledge-base",
        alpha: 0.7,
        top_k: 3,
      };

      try {
        const res = await fetch("/api/chat/message", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const err = await res.json();
          appendAssistantBubble("Error: " + (err.detail || res.statusText));
          return;
        }

        const data = await res.json();
        currentSessionId = data.session_id;
        document.getElementById("chat-session-id").textContent = `Session: ${currentSessionId.slice(0, 12)}...`;

        const lastMsg = data.messages[data.messages.length - 1];
        appendAssistantBubble(
          lastMsg.content,
          lastMsg.citations,
          lastMsg.guardrails,
          data.reformulated_query !== userText ? data.reformulated_query : null
        );
      } catch (err) {
        appendAssistantBubble("Connection error: " + err.message);
      } finally {
        sendBtn.disabled = false;
      }
    });
  }

  // Clear Chat Button
  const btnClearChat = document.getElementById("btn-clear-chat");
  if (btnClearChat) {
    btnClearChat.addEventListener("click", async () => {
      if (!currentSessionId) return;
      await fetch(`/api/chat/sessions/${currentSessionId}`, { method: "DELETE" });
      currentSessionId = null;
      document.getElementById("chat-session-id").textContent = "Session: Cleared";
      document.getElementById("chat-messages-feed").innerHTML = `
        <div class="chat-bubble assistant-bubble">
          <div class="bubble-header">
            <strong>Pinecone AI Assistant</strong>
            <span class="timestamp">Cleared</span>
          </div>
          <div class="bubble-content">
            Session cleared. You can start a fresh conversation whenever you are ready!
          </div>
        </div>
      `;
    });
  }

  // Refresh Catalog Button
  const btnRefreshCatalog = document.getElementById("btn-refresh-catalog");
  if (btnRefreshCatalog) {
    btnRefreshCatalog.addEventListener("click", fetchCatalog);
  }

  // Global Reset Button
  const btnReset = document.getElementById("btn-global-reset");
  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      if (!confirm("Are you sure you want to reset all Pinecone vectors across all namespaces?")) {
        return;
      }
      await fetch("/api/system/reset", { method: "POST" });
      alert("Pinecone vector index has been completely purged.");
      fetchCatalog();
      fetchTelemetry();
    });
  }
}

// Render Search Results
function renderSearchResults(matches) {
  const container = document.getElementById("search-results-box");
  if (!matches || matches.length === 0) {
    container.innerHTML = '<div class="empty-state">No matching vectors found for this query and namespace.</div>';
    return;
  }

  container.innerHTML = "";
  matches.forEach((m, idx) => {
    const meta = m.metadata || {};
    const scorePct = Math.min(100, Math.max(0, Math.round(m.score * 100)));

    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `
      <div class="result-card-header">
        <div>
          <strong>#${idx + 1} — ${meta.title || m.id}</strong>
          <span class="badge badge-outline" style="margin-left: 0.5rem;">${meta.category || 'General'}</span>
        </div>
        <div class="badge badge-vectors">Score: ${(m.score).toFixed(4)}</div>
      </div>
      <div class="result-score-bar-container">
        <div class="result-score-bar" style="width: ${scorePct}%;"></div>
      </div>
      <div class="result-text">${meta.text || 'No text snippet available.'}</div>
    `;
    container.appendChild(card);
  });
}

// Chat UI Bubbles
function appendUserBubble(text) {
  const feed = document.getElementById("chat-messages-feed");
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble user-bubble";
  bubble.innerHTML = `
    <div class="bubble-header">
      <strong>You</strong>
      <span class="timestamp">Just now</span>
    </div>
    <div class="bubble-content">${escapeHtml(text)}</div>
  `;
  feed.appendChild(bubble);
  feed.scrollTop = feed.scrollHeight;
}

function appendAssistantBubble(content, citations, guardrails, reformulatedQuery) {
  const feed = document.getElementById("chat-messages-feed");
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble assistant-bubble";

  let guardrailBadgeHtml = "";
  if (guardrails) {
    const isGrounded = guardrails.is_grounded;
    const badgeClass = isGrounded ? "badge-grounded" : "badge-ungrounded";
    const statusText = isGrounded ? "Grounded" : "Potential Hallucination";
    guardrailBadgeHtml = `
      <span class="badge ${badgeClass}" title="Context relevance: ${guardrails.context_relevance_score}">
        Faithfulness: ${Math.round(guardrails.faithfulness_score * 100)}% (${statusText})
      </span>
    `;
  }

  let reformHtml = "";
  if (reformulatedQuery) {
    reformHtml = `
      <div class="bubble-reformulation">
        🔍 Reformulated context query: "${escapeHtml(reformulatedQuery)}"
      </div>
    `;
  }

  let citationsHtml = "";
  if (citations && citations.length > 0) {
    const pills = citations.map((c) => `
      <div class="citation-pill">
        <div class="citation-pill-header">
          <span>📚 ${escapeHtml(c.title)} (Chunk #${c.chunk_index})</span>
          <span class="badge badge-tech">Similarity: ${(c.similarity_score).toFixed(3)}</span>
        </div>
        <div class="citation-snippet">"${escapeHtml(c.snippet)}"</div>
      </div>
    `).join("");

    citationsHtml = `
      <div class="citations-box">
        <div class="citations-title">Source Citations (${citations.length})</div>
        ${pills}
      </div>
    `;
  }

  bubble.innerHTML = `
    <div class="bubble-header">
      <strong>Pinecone AI Assistant</strong>
      ${guardrailBadgeHtml}
    </div>
    ${reformHtml}
    <div class="bubble-content">${escapeHtml(content)}</div>
    ${citationsHtml}
  `;
  feed.appendChild(bubble);
  feed.scrollTop = feed.scrollHeight;
}

// Fetch Catalog
async function fetchCatalog() {
  try {
    const res = await fetch("/api/documents");
    if (!res.ok) return;
    const docs = await res.json();
    renderCatalog(docs);
  } catch (err) {
    console.error("Failed to fetch catalog:", err);
  }
}

function renderCatalog(docs) {
  const container = document.getElementById("catalog-container");
  if (!docs || docs.length === 0) {
    container.innerHTML = '<div class="empty-state">No documents ingested yet. Submit one from the studio to start!</div>';
    return;
  }

  container.innerHTML = "";
  docs.forEach((doc) => {
    const item = document.createElement("div");
    item.className = "catalog-item";
    item.innerHTML = `
      <div class="catalog-info">
        <h4>${escapeHtml(doc.title)}</h4>
        <div class="catalog-meta">
          <span>Namespace: <strong>${doc.namespace}</strong></span>
          <span>•</span>
          <span>Category: ${doc.category}</span>
          <span>•</span>
          <span>Chunks: ${doc.chunk_count}</span>
          <span>•</span>
          <span>Chars: ${doc.character_count}</span>
        </div>
      </div>
      <button class="btn btn-outline-danger btn-sm" onclick="deleteDoc('${doc.id}')">Delete</button>
    `;
    container.appendChild(item);
  });
}

async function deleteDoc(docId) {
  if (!confirm(`Delete document ${docId} and all associated Pinecone vector chunks?`)) {
    return;
  }
  await fetch(`/api/documents/${docId}`, { method: "DELETE" });
  fetchCatalog();
  fetchTelemetry();
}
window.deleteDoc = deleteDoc;

// Fetch Telemetry Stats
async function fetchTelemetry() {
  try {
    const res = await fetch("/api/system/stats");
    if (!res.ok) return;
    const stats = await res.json();

    document.getElementById("stat-total-vectors").textContent = stats.total_vector_count;
    document.getElementById("top-vector-count").textContent = `${stats.total_vector_count} Vectors`;
    document.getElementById("stat-dimension").textContent = stats.dimension;
    document.getElementById("stat-metric").textContent = stats.metric.toUpperCase();

    const tbody = document.getElementById("telemetry-namespaces-body");
    const nsKeys = Object.keys(stats.namespaces || {});

    if (nsKeys.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3" class="text-muted">No namespaces populated yet.</td></tr>';
    } else {
      tbody.innerHTML = nsKeys.map((ns) => `
        <tr>
          <td><strong>${ns}</strong></td>
          <td>${stats.namespaces[ns].vector_count}</td>
          <td><span class="badge badge-vectors">Active</span></td>
        </tr>
      `).join("");
    }
  } catch (err) {
    console.error("Failed to fetch telemetry:", err);
  }
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
