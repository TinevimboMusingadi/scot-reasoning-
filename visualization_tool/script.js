let fileHandle = null;
let traces = [];
let currentIndex = -1;
let isOriginalGSM = false; // flag to handle differently shaped objects

// Theme Toggle
const themeToggleBtn = document.getElementById('theme-toggle');
themeToggleBtn.addEventListener('click', () => {
    const htmlEl = document.documentElement;
    if (htmlEl.getAttribute('data-theme') === 'dark') {
        htmlEl.setAttribute('data-theme', 'light');
        themeToggleBtn.textContent = 'Switch to Dark Mode';
    } else {
        htmlEl.setAttribute('data-theme', 'dark');
        themeToggleBtn.textContent = 'Switch to Light Mode';
    }
});

// UI Elements
const btnOpen = document.getElementById('btn-open');
const btnSaveDisk = document.getElementById('btn-save-disk');
const btnUpdateMem = document.getElementById('btn-update-mem');
const itemList = document.getElementById('item-list');
const problemText = document.getElementById('problem-text');
const traceEditor = document.getElementById('trace-editor');
const previewContent = document.getElementById('preview-content');
const toast = document.getElementById('toast');
const dropZone = document.getElementById('drop-zone');

const TAGS = ['meta_reasoning', 'decompose', 'deduction', 'abduction', 'induction', 'analogy', 'causal', 'think'];

// HARDCODED PATH LOADING (Fetch)
async function loadHardcoded(path) {
    try {
        if (window.location.protocol === 'file:') {
            throw new Error("SECURITY BLOCK: Browsers block local file fetching.\n\nSince you opened this via double-click, you must either:\n1. Drag & Drop the file onto the screen\n2. Use the 'Open File Dialog' button\n3. Run a local server ('python -m http.server') to use these buttons.");
        }

        fileHandle = null; // We didn't open this via picker, so no handle
        btnSaveDisk.disabled = true; // Can't easily save directly without a handle

        const response = await fetch(path);
        if (!response.ok) throw new Error("Could not fetch " + path + " (are you running a local web server?)");
        const contents = await response.text();
        
        parseJSONL(contents);
        showToast(`Loaded ${path}`);
    } catch (err) {
        alert(err.message);
    }
}

// DIALOG OPENING
btnOpen.addEventListener('click', async () => {
    try {
        if (window.showOpenFilePicker) {
            [fileHandle] = await window.showOpenFilePicker({
                types: [{ description: 'JSON Lines', accept: {'text/plain': ['.jsonl']} }]
            });
            const file = await fileHandle.getFile();
            const contents = await file.text();
            parseJSONL(contents);
            btnSaveDisk.disabled = false;
        } else {
            alert("File System Access API not supported in your browser.");
        }
    } catch (err) {
        console.warn("Cancelled open or failed:", err);
    }
});

// DRAG AND DROP LOGIC
document.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
});
document.addEventListener('drop', async (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
        const item = e.dataTransfer.items[0];
        if (item.kind === 'file') {
            const file = item.getAsFile();
            // Using modern API to keep handle if available
            if (item.getAsFileSystemHandle) {
                fileHandle = await item.getAsFileSystemHandle();
                btnSaveDisk.disabled = false;
            } else {
                fileHandle = null;
                btnSaveDisk.disabled = true; // Can't write back without handle
            }
            const contents = await file.text();
            parseJSONL(contents);
            showToast(`Loaded ${file.name}`);
        }
    }
});

function parseJSONL(contents) {
    traces = contents.split('\n').filter(l => l.trim()).map(line => JSON.parse(line));
    
    // Check if this is the original GSM dataset (has 'question' instead of 'problem' / 'scot_trace')
    if (traces.length > 0 && traces[0].question !== undefined) {
        isOriginalGSM = true;
    } else {
        isOriginalGSM = false;
    }

    renderList();
}

function renderList() {
    itemList.innerHTML = '';
    traces.forEach((t, i) => {
        const el = document.createElement('button');
        el.className = 'ghost-btn';
        // Show question snippet if original GSM, else just numbering
        let teaser = isOriginalGSM ? `[${i+1}] ${t.question.substring(0, 20)}...` : `Sample Trace 0${i+1}`;
        el.textContent = teaser;
        el.onclick = () => selectTrace(i);
        itemList.appendChild(el);
    });
}

function selectTrace(index) {
    if(currentIndex >= 0 && itemList.children[currentIndex]) {
        itemList.children[currentIndex].classList.remove('active');
    }
    currentIndex = index;
    itemList.children[currentIndex].classList.add('active');

    const t = traces[index];
    
    if (isOriginalGSM) {
        problemText.textContent = t.question;
        traceEditor.value = t.answer || "";
    } else {
        problemText.textContent = t.problem;
        traceEditor.value = t.scot_trace !== undefined ? t.scot_trace : (t.flat_trace || "");
    }
    
    btnUpdateMem.disabled = false;
    document.getElementById('btn-save-disk').disabled = !fileHandle;
    renderPreview(traceEditor.value);
}

traceEditor.addEventListener('input', () => {
    renderPreview(traceEditor.value);
});

btnUpdateMem.addEventListener('click', () => {
    if(currentIndex < 0) return;
    const text = traceEditor.value;
    
    if (isOriginalGSM) {
        traces[currentIndex].answer = text;
    } else {
        if (traces[currentIndex].scot_trace !== undefined) {
            traces[currentIndex].scot_trace = text;
        } else if (traces[currentIndex].flat_trace !== undefined) {
            traces[currentIndex].flat_trace = text;
        } else {
            traces[currentIndex].scot_trace = text;
        }
        const ansMatch = text.match(/<answer>(.*?)<\/answer>/s);
        if(ansMatch) traces[currentIndex].answer = ansMatch[1].trim();
    }

    showToast("Synchronized with Memory");
});

btnSaveDisk.addEventListener('click', async () => {
    if(!fileHandle || traces.length === 0) return;
    try {
        const writable = await fileHandle.createWritable();
        const blob = new Blob([traces.map(t => JSON.stringify(t)).join('\n') + '\n'], {type: 'text/plain'});
        await writable.write(blob);
        await writable.close();
        showToast("Disk Commit Successful");
    } catch (err) {
        
        // Fallback attempt to download if write fails
        const a = document.createElement('a');
        const blob = new Blob([traces.map(t => JSON.stringify(t)).join('\n') + '\n'], {type: 'text/plain'});
        a.href = URL.createObjectURL(blob);
        a.download = "edited_traces.jsonl";
        a.click();
    }
});

function renderPreview(text) {
    if (isOriginalGSM && !text.includes('<reasoning>')) {
        // If it's pure text (GSM answer), just render the text
        previewContent.innerHTML = escapeHtml(text);
        return;
    }

    let html = "";
    const regex = /<(\w+)>(.*?)<\/\1>/gs;
    let lastIdx = 0;
    let match;

    while((match = regex.exec(text)) !== null) {
        const pre = text.substring(lastIdx, match.index).trim();
        if(pre && pre !== '<reasoning>' && pre !== '</reasoning>') {
            html += `<div style="margin-bottom:15px;">${escapeHtml(pre)}</div>`;
        }

        const tag = match[1];
        const content = match[2].trim();
        
        if (tag === 'reasoning') {
            // skip wrapper
        } else if (TAGS.includes(tag)) {
            html += `
                <div class="reasoning-block block-${tag}">
                    <strong>${tag.replace('_', ' ')}</strong>
                    ${escapeHtml(content).replace(/\\n/g, '<br>')}
                </div>
            `;
        } else if (tag === 'answer') {
             html += `
                <div style="border: 1px solid var(--border-color); padding: 15px; text-align: center; font-weight: 600; margin-top:20px;">
                    ${escapeHtml(content)}
                </div>
            `;
        } else {
            html += `<div style="margin-bottom:15px;">${escapeHtml(match[0])}</div>`;
        }
        lastIdx = regex.lastIndex;
    }

    if (html === "") {
        if (text.includes('<reasoning>')) {
            const inner = text.replace(/<reasoning>/g, '').replace(/<\/reasoning>/g, '');
            html = parseInner(inner);
        } else {
            html = escapeHtml(text).replace(/\\n/g, '<br>');
        }
    } else if (text.includes('<reasoning>') && match && match[1] === 'reasoning') {
         html = parseInner(match[2]);
    }

    previewContent.innerHTML = html;
}

function parseInner(text) {
    let html = "";
    const regex = /<(\w+)>(.*?)<\/\1>/gs;
    let lastIdx = 0;
    let match;
    while((match = regex.exec(text)) !== null) {
        const tag = match[1];
        const content = match[2].trim();
        if (TAGS.includes(tag)) {
            html += `
                <div class="reasoning-block block-${tag}">
                    <strong>${tag.replace('_', ' ')}</strong>
                    ${escapeHtml(content).replace(/\n/g, '<br>')}
                </div>
            `;
        } else if (tag === 'answer') {
             html += `
                <div style="border: 1px solid var(--border-color); padding: 15px; text-align: center; font-weight: 600; margin-top:20px;">
                    ${escapeHtml(content)}
                </div>
            `;
        } else {
            html += `<div style="margin-bottom:15px;">${escapeHtml(match[0])}</div>`;
        }
        lastIdx = regex.lastIndex;
    }
    if(lastIdx < text.length) {
         const trail = text.substring(lastIdx).trim();
         if(trail) html += `<div style="margin-bottom:15px;">${escapeHtml(trail)}</div>`;
    }
    return html || escapeHtml(text);
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), 3000);
}
