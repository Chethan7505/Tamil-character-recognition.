document.addEventListener("DOMContentLoaded", function () {
    suggest();
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");

    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function draw(e) {
        if (!isDrawing) return;
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
        [lastX, lastY] = [e.offsetX, e.offsetY];
    }

    canvas.addEventListener("mousedown", (e) => {
        isDrawing = true;
        [lastX, lastY] = [e.offsetX, e.offsetY];
    });
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", () => isDrawing = false);
    canvas.addEventListener("mouseout", () => isDrawing = false);
});

function submitImage() {
    const canvas = document.getElementById("canvas");
    var img = canvas.toDataURL();
    predictImage(img);
    suggest();
}

function predictImage(img) {
    fetch("/predict", { method: "POST", body: img })
    .then(resp => resp.text())
    .then(data => {
        console.log("Prediction response:", data);
        var strings = data.split(" ");
        var character = strings[0];
        var prob = strings[1];

        const idleEl = document.getElementById("result-idle");
        if (idleEl) idleEl.style.display = "none";

        const guessEl = document.getElementById("guess");
        guessEl.textContent = character;
        guessEl.classList.remove("pop");
        void guessEl.offsetWidth;
        guessEl.classList.add("pop");

        const confidenceEl = document.getElementById("confidence");
        confidenceEl.textContent = "(confidence: " + prob + "%)";
    })
    .catch(err => console.error("Prediction error:", err));
}

function clearCanvas() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const guessEl = document.getElementById("guess");
    const confidenceEl = document.getElementById("confidence");
    const idleEl = document.getElementById("result-idle");
    if (guessEl) guessEl.textContent = "";
    if (confidenceEl) confidenceEl.textContent = "";
    if (idleEl) idleEl.style.display = "flex";
}

function suggest() {
    const suggestion = document.getElementById("suggestion");
    if (!suggestion) return;
    fetch("/suggest")
        .then(r => r.text())
        .then(data => {
            suggestion.textContent = "Draw a Tamil character. Not sure what to draw? Try " + data + ".";
        });
}

// ── Upload preview fix ──
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) showPreview(file);
}

function showPreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const dropZone  = document.getElementById("drop-zone");
        const dropInner = document.getElementById("drop-inner");
        const previewImg = document.getElementById("preview-img");

        // Show the image, hide the placeholder text
        previewImg.src = e.target.result;

        // Apply styles directly so there's no CSS conflict
        previewImg.style.cssText = `
            display: block;
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            object-fit: contain;
            border-radius: 14px;
            z-index: 2;
        `;

        // Hide placeholder text below the image
        if (dropInner) dropInner.style.display = "none";
    };
    reader.readAsDataURL(file);
}

function clearUpload() {
    document.getElementById("file-input").value = "";
    const previewImg = document.getElementById("preview-img");
    previewImg.src = "";
    previewImg.style.display = "none";

    const dropInner = document.getElementById("drop-inner");
    if (dropInner) dropInner.style.display = "flex";

    const guessEl = document.getElementById("guess");
    const confidenceEl = document.getElementById("confidence");
    const idleEl = document.getElementById("result-idle");
    if (guessEl) guessEl.textContent = "";
    if (confidenceEl) confidenceEl.textContent = "";
    if (idleEl) idleEl.style.display = "flex";
}

function submitUpload() {
    const fi = document.getElementById("file-input");
    if (!fi.files[0]) { alert("Please select an image first!"); return; }

    const sp = document.getElementById("upload-spinner");
    if (sp) sp.style.display = "block";

    const idleEl = document.getElementById("result-idle");
    if (idleEl) idleEl.style.display = "none";

    const guessEl = document.getElementById("guess");
    const confidenceEl = document.getElementById("confidence");
    if (guessEl) guessEl.textContent = "";
    if (confidenceEl) confidenceEl.textContent = "";

    const fd = new FormData();
    fd.append("image", fi.files[0]);

    fetch("/predict_upload", { method: "POST", body: fd })
    .then(r => r.text())
    .then(res => {
        if (sp) sp.style.display = "none";
        const p = res.split("|");
        if (p.length === 2) {
            if (idleEl) idleEl.style.display = "none";
            if (guessEl) {
                guessEl.textContent = p[0];
                guessEl.classList.remove("pop");
                void guessEl.offsetWidth;
                guessEl.classList.add("pop");
            }
            if (confidenceEl) confidenceEl.textContent = "(confidence: " + p[1] + "%)";
        } else {
            if (guessEl) guessEl.textContent = res;
        }
    })
    .catch(() => {
        if (sp) sp.style.display = "none";
        if (guessEl) guessEl.textContent = "Error!";
    });
}

// Drag and drop
document.addEventListener("DOMContentLoaded", function() {
    const dz = document.getElementById("drop-zone");
    if (!dz) return;
    dz.addEventListener("dragover",  e => { e.preventDefault(); dz.classList.add("drag-over"); });
    dz.addEventListener("dragleave", ()  => dz.classList.remove("drag-over"));
    dz.addEventListener("drop", e => {
        e.preventDefault();
        dz.classList.remove("drag-over");
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith("image/")) showPreview(f);
    });
});