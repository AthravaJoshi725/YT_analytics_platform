// --- Configuration ---
// Update this if your backend runs on a different host/port or behind Docker.
const API_BASE_URL = 'http://127.0.0.1:8000';

// --- DOM references ---
const linkForm = document.getElementById('link-form');
const youtubeInput = document.getElementById('youtube-link');
const analyzeBtn = document.getElementById('analyze-btn');

const statusPanel = document.getElementById('status-panel');
const statusTitle = document.getElementById('status-title');
const statusText = document.getElementById('status-text');
const statusError = document.getElementById('status-error');

const videoDetailsEl = document.getElementById('video-details');
const videoThumb = document.getElementById('video-thumbnail');
const videoTitle = document.getElementById('video-title');
const videoChannel = document.getElementById('video-channel');
const videoDate = document.getElementById('video-date');
const videoDesc = document.getElementById('video-description');

const chatCard = document.getElementById('chat-card');
const chatStatusDot = document.getElementById('chat-status-dot');
const chatStatusText = document.getElementById('chat-status-text');
const chatHistory = document.getElementById('chat-history');
const chatForm = document.getElementById('chat-form');
const userQueryInput = document.getElementById('user-query');
const sendBtn = document.getElementById('send-btn');

// --- State ---
let currentYoutubeLink = '';
let ragReady = false;
let ragPollIntervalId = null;

// --- Helpers ---
const showStatus = (title, text) => {
    statusTitle.textContent = title;
    statusText.textContent = text;
    statusPanel.classList.remove('hidden');
    statusError.classList.add('hidden');
};

const showError = (message) => {
    statusError.textContent = message;
    statusError.classList.remove('hidden');
};

const setLoading = (isLoading) => {
    analyzeBtn.disabled = isLoading;
    youtubeInput.disabled = isLoading;
    if (isLoading) {
        showStatus(
            'Preparing RAG index…',
            'Fetching comments and building embeddings. This can take a bit for long videos.'
        );
    }
};

const enableChat = () => {
    ragReady = true;
    if (ragPollIntervalId) {
        clearInterval(ragPollIntervalId);
        ragPollIntervalId = null;
    }
    chatCard.classList.remove('disabled');
    chatStatusDot.classList.add('ready');
    chatStatusText.textContent = 'RAG is ready. Ask anything about the comments.';
    userQueryInput.disabled = false;
    sendBtn.disabled = false;
};

const addMessage = (type, text) => {
    const msg = document.createElement('div');
    msg.className = `chat-message ${type}`;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    msg.appendChild(bubble);
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
};

// --- API calls ---
const callAnalyze = async (youtubeLink) => {
    const url = `${API_BASE_URL}/analyze?youtube_link=${encodeURIComponent(
        youtubeLink
    )}`;
    const res = await fetch(url, { method: 'POST' });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Failed to analyze video (status ${res.status})`);
    }
    return res.json(); // video_details dict
};

const callAsk = async (youtubeLink, userQuery) => {
    const url = `${API_BASE_URL}/ask?youtube_link=${encodeURIComponent(
        youtubeLink
    )}&user_query=${encodeURIComponent(userQuery)}`;
    const res = await fetch(url, { method: 'POST' });

    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Failed to ask question (status ${res.status})`);
    }

    // Backend returns either a plain string or {"error": "..."}
    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
        const data = await res.json();
        if (typeof data === 'string') {
            return data;
        }
        if (data.error) {
            throw new Error(data.error);
        }
        return JSON.stringify(data);
    }

    return res.text();
};

// Poll /ask with a harmless test query until RAG is ready.
const startRagPolling = () => {
    if (!currentYoutubeLink) return;
    if (ragPollIntervalId) {
        clearInterval(ragPollIntervalId);
    }

    ragPollIntervalId = setInterval(async () => {
        if (ragReady || !currentYoutubeLink) {
            clearInterval(ragPollIntervalId);
            ragPollIntervalId = null;
            return;
        }
        try {
            await callAsk(currentYoutubeLink, 'RAG readiness check');
            // If we get here without throwing "Rag not ready yet", RAG is ready.
            enableChat();
            showStatus(
                'RAG ready',
                'You can now start asking questions about the comments.'
            );
        } catch (err) {
            const msg = err.message || '';
            if (msg.includes('Rag not ready yet')) {
                // Still building; keep polling silently.
                return;
            }
            // Any other error: surface it and stop polling.
            console.error('RAG polling error:', err);
            showError(msg || 'Error while checking RAG readiness.');
            if (ragPollIntervalId) {
                clearInterval(ragPollIntervalId);
                ragPollIntervalId = null;
            }
        }
    }, 5000); // poll every 5 seconds
};

// --- Event handlers ---
linkForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const link = youtubeInput.value.trim();
    if (!link) return;

    // Reset state
    currentYoutubeLink = link;
    ragReady = false;
    videoDetailsEl.classList.add('hidden');
    chatHistory.innerHTML = '';
    chatCard.classList.add('disabled');
    chatStatusDot.classList.remove('ready');
    chatStatusText.textContent = 'Building RAG index for this video…';
    userQueryInput.disabled = true;
    sendBtn.disabled = true;
    addMessage(
        'ai',
        'Working on this video. I am fetching comments and building a RAG index so I can answer detailed questions.'
    );

    setLoading(true);

    try {
        const details = await callAnalyze(link);

        // Update video details panel
        if (details.thumbnail) {
            videoThumb.src = details.thumbnail;
            videoThumb.classList.remove('hidden');
        } else {
            videoThumb.classList.add('hidden');
        }
        videoTitle.textContent = details.title || 'Untitled video';
        videoChannel.textContent = details.channelName || 'Unknown channel';
        videoDate.textContent = details.publishedAt
            ? new Date(details.publishedAt).toLocaleDateString()
            : '';
        videoDesc.textContent = details.description
            ? `${details.description.slice(0, 220)}…`
            : 'No description available.';

        videoDetailsEl.classList.remove('hidden');

        // Start polling /ask to detect when RAG is ready.
        showStatus(
            'Building RAG index…',
            'Comments have been fetched. Waiting for the vector index to finish building.'
        );
        startRagPolling();
    } catch (err) {
        console.error(err);
        showError(err.message || 'Failed to analyze video.');
        addMessage('ai', `Error: ${err.message || 'Failed to analyze video.'}`);
    } finally {
        setLoading(false);
    }
});

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = userQueryInput.value.trim();
    if (!query || !ragReady || !currentYoutubeLink) return;

    addMessage('user', query);
    userQueryInput.value = '';

    const loadingMsgText = 'Thinking over the comments…';
    addMessage('ai', loadingMsgText);
    const loadingMsgEl = chatHistory.lastElementChild;

    try {
        const answer = await callAsk(currentYoutubeLink, query);
        loadingMsgEl.querySelector('.bubble').textContent = answer;
    } catch (err) {
        loadingMsgEl.querySelector('.bubble').textContent =
            `Error: ${err.message || 'Unable to get an answer.'}`;
    }
});


