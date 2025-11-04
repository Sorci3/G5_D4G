const textarea = document.getElementById("text");
const charCount = document.getElementById("charCount");
const copyBtn = document.getElementById("copyBtnOutput");
const summaryEl = document.getElementById("summary");
const metricsDiv = document.getElementById("metrics");
const summaryLabel = document.getElementById("summaryLabel");
const outputDiv = document.getElementById("output");
const btn = document.getElementById("btn");

function autoResize() {
    const minHeight = 200;
    textarea.style.height = "auto";
    textarea.style.height = Math.max(textarea.scrollHeight, minHeight) + "px";
}

textarea.addEventListener("input", () => {
    charCount.textContent = `${textarea.value.length}/4000 caractères`;
    autoResize();
    btn.disabled = !textarea.value.trim();
});

window.addEventListener("load", autoResize);

copyBtn.style.display = "none";
metricsDiv.style.display = "none";
summaryLabel.style.display = "none";
outputDiv.style.display = "none";

btn.addEventListener("click", async () => {
    const text = textarea.value;
    const optimized = document.getElementById("optimized").checked;

    if (!text.trim()) {
        summaryEl.textContent = "";
        copyBtn.style.display = "none";
        metricsDiv.style.display = "none";
        summaryLabel.style.display = "none";
        outputDiv.style.display = "none";
        return;
    }

    try {
        const res = await fetch("/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, optimized })
        });

        const data = await res.json();

        if (data.error) {
            summaryEl.textContent = "Erreur : " + data.error;
            copyBtn.style.display = "none";
            metricsDiv.style.display = "none";
            summaryLabel.style.display = "none";
            outputDiv.style.display = "none";
            return;
        }

        const result = data;

        const hasSummary = result.summary && result.summary.trim() !== "";

        summaryEl.textContent = result.summary;

        copyBtn.style.display = hasSummary ? "block" : "none";
        metricsDiv.style.display = hasSummary ? "block" : "none";
        summaryLabel.style.display = hasSummary ? "block" : "none";
        outputDiv.style.display = hasSummary ? "block" : "none";

        if (hasSummary) {
            document.getElementById("time").textContent = result.latency_ms.toFixed(1);
            document.getElementById("energy").textContent = result.energy_wh.toFixed(6);
        }

    } catch (error) {
        summaryEl.textContent = "Erreur lors de la requête : " + error;
        copyBtn.style.display = "none";
        metricsDiv.style.display = "none";
        summaryLabel.style.display = "none";
        outputDiv.style.display = "none";
    }
});

copyBtn.addEventListener("click", () => {
    const summary = summaryEl.textContent;

    if (!summary) {
        return;
    }

    navigator.clipboard.writeText(summary)
        .catch(() => {});
});

btn.disabled = !textarea.value.trim();