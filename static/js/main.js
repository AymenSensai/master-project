document.addEventListener('DOMContentLoaded', () => {
    loadIdentities();
    const visUpload = document.getElementById('vis-upload');
    const nirUpload = document.getElementById('nir-upload');
    const visInput = document.getElementById('vis-input');
    const nirInput = document.getElementById('nir-input');
    const visPreview = document.getElementById('vis-preview');
    const nirPreview = document.getElementById('nir-preview');
    const verifyBtn = document.getElementById('verify-btn');
    const resultContainer = document.getElementById('result-container');
    const resultStatus = document.getElementById('result-status');
    const similarityFill = document.getElementById('similarity-fill');
    const similarityValue = document.getElementById('similarity-value');
    const resultMessage = document.getElementById('result-message');
    const btnLoader = verifyBtn.querySelector('.loader');
    const btnText = verifyBtn.querySelector('.btn-text');

    const modeGallery = document.getElementById('mode-gallery');
    const gallerySection = document.getElementById('gallery-section');
    const galleryGrid = document.getElementById('gallery-grid');

    const modeVerify = document.getElementById('mode-verify');
    const modeRecognize = document.getElementById('mode-recognize');
    const modeLive = document.getElementById('mode-live');
    const modeChallenge = document.getElementById('mode-challenge');
    const modeDashboard = document.getElementById('mode-dashboard');
    const challengeSection = document.getElementById('challenge-section');
    const identityList = document.getElementById('identity-list');
    const faceCanvas = document.getElementById('face-canvas');

    const dashboardSection = document.getElementById('dashboard-section');
    const statIdentities = document.getElementById('stat-identities');
    const statVis = document.getElementById('stat-vis');
    const statNir = document.getElementById('stat-nir');
    const statDevice = document.getElementById('stat-device');

    const xaiSection = document.getElementById('xai-section');
    const xaiOriginal = document.getElementById('xai-original');
    const xaiHeatmap = document.getElementById('xai-heatmap');
    const webcamSection = document.getElementById('webcam-section');
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const startWebcamBtn = document.getElementById('start-webcam');
    const stopWebcamBtn = document.getElementById('stop-webcam');
    const hudStatus = document.getElementById('hud-status');
    const hudLatency = document.getElementById('hud-latency');

    let visFile = null;
    let nirFile = null;
    let currentMode = 'verify';
    let webcamStream = null;
    let isLiveProcessing = false;
    let scoreBuffer = [];
    const BUFFER_SIZE = 6;

    // Mode Switching
    modeVerify.addEventListener('click', () => switchMode('verify'));
    modeRecognize.addEventListener('click', () => switchMode('recognize'));
    modeLive.addEventListener('click', () => switchMode('live'));
    modeGallery.addEventListener('click', () => switchMode('gallery'));
    modeChallenge.addEventListener('click', () => switchMode('challenge'));
    modeDashboard.addEventListener('click', () => switchMode('dashboard'));

    startWebcamBtn.addEventListener('click', startWebcam);
    stopWebcamBtn.addEventListener('click', stopWebcam);

    function switchMode(mode) {
        currentMode = mode;
        document.body.className = `${mode}-mode`;

        modeVerify.classList.toggle('active', mode === 'verify');
        modeRecognize.classList.toggle('active', mode === 'recognize');
        modeLive.classList.toggle('active', mode === 'live');
        modeGallery.classList.toggle('active', mode === 'gallery');
        modeChallenge.classList.toggle('active', mode === 'challenge');
        modeDashboard.classList.toggle('active', mode === 'dashboard');

        // Reset images and results
        visFile = null;
        nirFile = null;
        visInput.value = '';
        nirInput.value = '';
        visPreview.hidden = true;
        visPreview.src = '';
        nirPreview.hidden = true;
        nirPreview.src = '';
        if (faceCanvas) faceCanvas.hidden = true;
        visUpload.querySelector('.upload-placeholder').style.opacity = '';
        nirUpload.querySelector('.upload-placeholder').style.opacity = '';
        resultContainer.hidden = true;

        const comparisonGrid = document.querySelector('.comparison-grid');
        const actionSection = document.querySelector('.action-section');
        const galleryPeek = document.querySelector('.gallery-peek');
        const resultsContainer = document.getElementById('results-container');

        const hideMain = mode === 'gallery' || mode === 'dashboard' || mode === 'live' || mode === 'challenge';
        btnText.textContent = hideMain ? '' : (mode === 'verify' ? 'VÉRIFIER L\'IDENTITÉ' : 'RECONNAÎTRE LE VISAGE');

        if (comparisonGrid) comparisonGrid.style.display = hideMain ? 'none' : '';
        if (actionSection) actionSection.style.display = hideMain ? 'none' : '';
        if (galleryPeek) galleryPeek.style.display = hideMain ? 'none' : '';
        if (resultsContainer && hideMain) resultsContainer.hidden = true;

        gallerySection.style.display = (mode === 'gallery') ? 'block' : 'none';
        dashboardSection.style.display = (mode === 'dashboard') ? 'block' : 'none';
        webcamSection.style.display = (mode === 'live') ? 'block' : 'none';
        challengeSection.style.display = (mode === 'challenge') ? 'block' : 'none';

        if (mode === 'recognize') {
            nirUpload.querySelector('p').innerHTML = 'Cliquez ou glissez l\'image<br><strong>NIR</strong>';
        } else {
            nirUpload.querySelector('p').innerHTML = 'Cliquez ou glissez l\'image<br>NIR';
        }

        if (mode !== 'live' && webcamStream) stopWebcam();

        if (mode === 'gallery') loadGallery();
        if (mode === 'recognize') loadIdentities();
        if (mode === 'dashboard') loadDashboard();
        if (mode === 'challenge') loadChallenge();

        checkReady();
    }

    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1280, height: 720, facingMode: "user" }
            });
            webcamVideo.srcObject = webcamStream;
            startWebcamBtn.hidden = true;
            stopWebcamBtn.hidden = false;
            hudStatus.textContent = "SCANNING";
            isLiveProcessing = true;
            processLiveFrame();
        } catch (err) {
            console.error("Webcam Error:", err);
            alert("Impossible d'accéder à la caméra. Vérifiez les permissions.");
        }
    }

    function stopWebcam() {
        isLiveProcessing = false;
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
        startWebcamBtn.hidden = false;
        stopWebcamBtn.hidden = true;
        hudStatus.textContent = "OFFLINE";
        const ctx = webcamCanvas.getContext('2d');
        ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
    }

    async function processLiveFrame() {
        if (!isLiveProcessing || currentMode !== 'live') return;

        const startTime = Date.now();
        const canvas = document.createElement('canvas');
        canvas.width = 640; // Smaller for speed
        canvas.height = 360;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            try {
                const response = await fetch('/api/recognize', { method: 'POST', body: formData });
                const data = await response.json();

                hudLatency.textContent = Date.now() - startTime;

                if (data.face_box) {
                    // Temporal Smoothing (Moving Average)
                    scoreBuffer.push(data.similarity);
                    if (scoreBuffer.length > BUFFER_SIZE) scoreBuffer.shift();
                    const avgSimilarity = scoreBuffer.reduce((a, b) => a + b, 0) / scoreBuffer.length;
                    
                    drawLiveHUD(data.face_box, data.identity, data.matched, avgSimilarity);
                } else {
                    scoreBuffer = []; // Reset smoothing if face is lost
                    const ctxW = webcamCanvas.getContext('2d');
                    ctxW.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
                }
            } catch (e) {
                console.error("Live process error:", e);
            }

            // Schedule next frame
            if (isLiveProcessing) {
                setTimeout(processLiveFrame, 800); // ~1.2 FPS is enough for demo
            }
        }, 'image/jpeg', 0.6);
    }

    function drawLiveHUD(box, identity, matched, similarity) {
        const ctx = webcamCanvas.getContext('2d');
        webcamCanvas.width = webcamVideo.clientWidth;
        webcamCanvas.height = webcamVideo.clientHeight;
        ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);

        const scaleX = webcamCanvas.width / box.img_w;
        const scaleY = webcamCanvas.height / box.img_h;
        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        const liveThreshold = 0.60;
        const isMatched = matched && (similarity >= liveThreshold);

        const color = isMatched ? '#00ffff' : '#ff3b3b';
        const shadow = isMatched ? 'rgba(0, 255, 255, 0.5)' : 'rgba(255, 59, 59, 0.5)';

        // Draw Target Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 5]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        // Draw Label
        ctx.fillStyle = color;
        ctx.font = 'bold 14px "Courier New"';
        const label = isMatched ? `${identity.toUpperCase()} [${(similarity * 100).toFixed(1)}%]` : "ID_UNKNOWN";
        ctx.fillText(label, x, y - 10);

        // Cyberpunk brackets
        ctx.lineWidth = 4;
        const bLen = 20;
        ctx.beginPath();
        // TL
        ctx.moveTo(x, y + bLen); ctx.lineTo(x, y); ctx.lineTo(x + bLen, y);
        // TR
        ctx.moveTo(x + w - bLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + bLen);
        // BL
        ctx.moveTo(x, y + h - bLen); ctx.lineTo(x, y + h); ctx.lineTo(x + bLen, y + h);
        // BR
        ctx.moveTo(x + w - bLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - bLen);
        ctx.stroke();
    }

    async function loadDashboard() {
        try {
            const response = await fetch('/api/stats');
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            statIdentities.textContent = data.identities;
            statVis.textContent = data.vis_images;
            statNir.textContent = data.nir_images;
            statDevice.textContent = data.device;
        } catch (error) {
            console.error('Error loading dashboard:', error);
        }
    }

    async function loadGallery() {
        try {
            galleryGrid.innerHTML = '<div class="loading-gallery">Chargement de la base de données...</div>';
            const response = await fetch('/api/gallery');
            const data = await response.json();

            galleryGrid.innerHTML = '';
            if (data.gallery.length === 0) {
                galleryGrid.innerHTML = '<div class="loading-gallery">Aucune identité trouvée dans la base de données.</div>';
                return;
            }
            data.gallery.forEach(item => {
                const card = document.createElement('div');
                card.className = 'gallery-item';
                card.innerHTML = `
                    <img src="${item.image_url}" alt="${item.identity}">
                    <div class="info"><span class="name">${item.identity}</span></div>
                `;
                galleryGrid.appendChild(card);
            });
        } catch (error) {
            console.error('Error loading gallery:', error);
            galleryGrid.innerHTML = '<div class="loading-gallery">Erreur lors du chargement de la galerie.</div>';
        }
    }

    async function loadIdentities() {
        try {
            const response = await fetch('/api/identities');
            const data = await response.json();
            identityList.innerHTML = '';
            if (data.identities.length === 0) {
                identityList.innerHTML = '<span class="identity-badge">La galerie est vide</span>';
            } else {
                data.identities.forEach(id => {
                    const badge = document.createElement('span');
                    badge.className = 'identity-badge';
                    badge.textContent = id;
                    identityList.appendChild(badge);
                });
            }
        } catch (error) {
            console.error('Error loading identities:', error);
        }
    }

    // Upload clicks
    visUpload.addEventListener('click', () => visInput.click());
    nirUpload.addEventListener('click', () => nirInput.click());

    visInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            visFile = file;
            handlePreview(file, visPreview, visUpload.querySelector('.upload-placeholder'));
            checkReady();
        }
    });

    nirInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            nirFile = file;
            handlePreview(file, nirPreview, nirUpload.querySelector('.upload-placeholder'));
            checkReady();
        }
    });

    function handlePreview(file, previewImg, placeholder) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.hidden = false;
            placeholder.style.opacity = '0';

            // Clear and hide face canvas on new upload
            if (faceCanvas) {
                faceCanvas.hidden = true;
                const ctx = faceCanvas.getContext('2d');
                ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
            }
        };
        reader.readAsDataURL(file);
    }

    function checkReady() {
        if (currentMode === 'verify') {
            verifyBtn.disabled = !(visFile && nirFile);
        } else {
            verifyBtn.disabled = !nirFile;
        }
    }

    verifyBtn.addEventListener('click', async () => {
        if (currentMode === 'verify') {
            await performVerification();
        } else {
            await performRecognition();
        }
    });

    async function performVerification() {
        if (!visFile || !nirFile) return;
        setLoading(true, 'ANALYSE EN COURS...');
        resultContainer.hidden = true;

        const formData = new FormData();
        formData.append('vis_image', visFile);
        formData.append('nir_image', nirFile);

        try {
            const response = await fetch('/api/compare', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.error) alert('Erreur : ' + data.error);
            else displayResult(data);
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Une erreur s\'est produite lors de la vérification.');
        } finally {
            setLoading(false, 'VÉRIFIER L\'IDENTITÉ');
        }
    }

    async function performRecognition() {
        if (!nirFile) return;
        setLoading(true, 'RECONNAISSANCE EN COURS...');
        resultContainer.hidden = true;

        const formData = new FormData();
        formData.append('image', nirFile);

        try {
            const response = await fetch('/api/recognize', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.error) {
                alert('Erreur: ' + data.error);
            } else if (data.message === 'Gallery is empty') {
                alert('La galerie est vide. Veuillez ajouter des images dans le dossier gallery/.');
            } else {
                displayRecognitionResult(data);
                if (data.face_box) drawFaceHighlight(data.face_box);
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Une erreur s\'est produite lors de la reconnaissance.');
        } finally {
            setLoading(false, 'RECONNAÎTRE LE VISAGE');
        }
    }

    function drawFaceHighlight(box) {
        const ctx = faceCanvas.getContext('2d');
        faceCanvas.width = nirPreview.clientWidth;
        faceCanvas.height = nirPreview.clientHeight;
        faceCanvas.hidden = false;

        const scaleX = faceCanvas.width / box.img_w;
        const scaleY = faceCanvas.height / box.img_h;
        const centerX = (box.x + box.w / 2) * scaleX;
        const centerY = (box.y + box.h / 2) * scaleY;
        const radius = (Math.max(box.w, box.h) / 2) * Math.min(scaleX, scaleY) * 1.2;

        ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 3;
        ctx.shadowBlur = 15;
        ctx.shadowColor = '#2563eb';
        ctx.stroke();
    }

    function setLoading(isLoading, text) {
        verifyBtn.disabled = isLoading;
        btnLoader.hidden = !isLoading;
        btnText.textContent = text;
        if (isLoading) faceCanvas.hidden = true;
    }

    function displayResult(data) {
        resultContainer.hidden = false;
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        resultContainer.classList.remove('match', 'mismatch');

        const simPercent = (data.similarity * 100).toFixed(2);

        if (data.match) {
            resultContainer.classList.add('match');
            resultStatus.textContent = 'IDENTITÉ CORRESPONDANTE';
            resultMessage.textContent = '';
            if (data.heatmap) showXAI(data.face_crop, data.heatmap);
        } else {
            resultContainer.classList.add('mismatch');
            resultStatus.textContent = 'IDENTITÉ NON CORRESPONDANTE';
            resultMessage.textContent = `Faible similarité détectée. Il semble s'agir de personnes différentes.`;
            xaiSection.hidden = true;
        }
        updateSimilarityBar(simPercent);
    }

    function displayRecognitionResult(data) {
        resultContainer.hidden = false;
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        resultContainer.classList.remove('match', 'mismatch');

        const simPercent = (data.similarity * 100).toFixed(2);

        if (data.matched) {
            resultContainer.classList.add('match');
            resultStatus.textContent = `RECONNU : ${data.identity}`;
            resultMessage.textContent = `La meilleure correspondance trouvée dans la galerie pour ${data.identity} avec ${simPercent}% de confiance.`;
            if (data.heatmap) showXAI(data.face_crop, data.heatmap);
        } else {
            resultContainer.classList.add('mismatch');
            resultStatus.textContent = 'IDENTITÉ INCONNUE';
            resultMessage.textContent = `Aucune correspondance fiable trouvée dans la galerie. La meilleure était ${data.identity} avec une faible confiance (${simPercent}%).`;
            xaiSection.hidden = true;
        }
        updateSimilarityBar(simPercent);
    }

    function showXAI(faceCropB64, heatmapB64) {
        xaiOriginal.src = faceCropB64
            ? `data:image/jpeg;base64,${faceCropB64}`
            : nirPreview.src;
        xaiHeatmap.src = `data:image/jpeg;base64,${heatmapB64}`;
        xaiSection.hidden = false;
    }

    function updateSimilarityBar(percent) {
        similarityValue.textContent = `${percent}%`;
        const displayWidth = Math.max(0, Math.min(100, percent));
        similarityFill.style.width = '0%';
        setTimeout(() => { similarityFill.style.width = `${displayWidth}%`; }, 50);
    }

    // Challenge (Défi IA) Mode
    const challengeProbeImg = document.getElementById('challenge-probe-img');
    const challengeCandidateImg = document.getElementById('challenge-candidate-img');
    const challengeChoices = document.getElementById('challenge-choices');
    const choiceSameBtn = document.getElementById('choice-same');
    const choiceDifferentBtn = document.getElementById('choice-different');
    const challengeReveal = document.getElementById('challenge-reveal');
    const revealCorrect = document.getElementById('reveal-correct');
    const revealAi = document.getElementById('reveal-ai');
    const revealUser = document.getElementById('reveal-user');
    const challengeNextBtn = document.getElementById('challenge-next-btn');
    const challengeLoading = document.getElementById('challenge-loading');
    const challengeError = document.getElementById('challenge-error');
    const challengeBoard = document.getElementById('challenge-board');
    const scoreUserEl = document.getElementById('score-user');
    const scoreAiEl = document.getElementById('score-ai');

    let currentChallenge = null;
    let challengeLocked = false;
    let scoreUser = 0;
    let scoreAi = 0;

    challengeNextBtn.addEventListener('click', loadChallenge);
    choiceSameBtn.addEventListener('click', () => onChoiceMade(true));
    choiceDifferentBtn.addEventListener('click', () => onChoiceMade(false));

    async function loadChallenge() {
        challengeLocked = false;
        currentChallenge = null;
        challengeError.hidden = true;
        challengeError.textContent = '';
        challengeReveal.hidden = true;
        challengeBoard.style.display = 'none';
        challengeLoading.hidden = false;
        challengeNextBtn.hidden = true;
        
        // Reset button states
        choiceSameBtn.disabled = false;
        choiceDifferentBtn.disabled = false;
        choiceSameBtn.classList.remove('is-correct', 'is-wrong', 'is-user-pick', 'is-ai-pick');
        choiceDifferentBtn.classList.remove('is-correct', 'is-wrong', 'is-user-pick', 'is-ai-pick');

        try {
            const response = await fetch('/api/challenge/new');
            const data = await response.json();
            if (data.error) {
                challengeError.textContent = data.error;
                challengeError.hidden = false;
                challengeLoading.hidden = true;
                return;
            }
            currentChallenge = data;
            renderChallenge(data);
        } catch (err) {
            console.error('Challenge fetch error:', err);
            challengeError.textContent = "Impossible de charger le défi.";
            challengeError.hidden = false;
        } finally {
            challengeLoading.hidden = true;
        }
    }

    function renderChallenge(data) {
        challengeBoard.style.display = '';
        challengeProbeImg.src = data.probe_url;
        challengeCandidateImg.src = data.candidate_url;
        challengeBoard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function onChoiceMade(userChoice) {
        if (challengeLocked || !currentChallenge) return;
        challengeLocked = true;

        const isSame = currentChallenge.is_same;
        const aiChoice = currentChallenge.ai_same;
        const userCorrect = userChoice === isSame;
        const aiCorrect = aiChoice === isSame;

        if (userCorrect) scoreUser++;
        if (aiCorrect) scoreAi++;
        scoreUserEl.textContent = scoreUser;
        scoreAiEl.textContent = scoreAi;

        // Visual feedback on buttons
        const correctBtn = isSame ? choiceSameBtn : choiceDifferentBtn;
        const wrongBtn = isSame ? choiceDifferentBtn : choiceSameBtn;
        const userBtn = userChoice ? choiceSameBtn : choiceDifferentBtn;
        const aiBtn = aiChoice ? choiceSameBtn : choiceDifferentBtn;

        choiceSameBtn.disabled = true;
        choiceDifferentBtn.disabled = true;

        correctBtn.classList.add('is-correct');
        userBtn.classList.add('is-user-pick');
        aiBtn.classList.add('is-ai-pick');
        
        if (!userCorrect) userBtn.classList.add('is-wrong');

        revealCorrect.textContent = isSame ? "Même personne" : "Personnes différentes";
        revealAi.textContent = `${aiChoice ? "Même" : "Différente"} ${aiCorrect ? '✓' : '✗'}`;
        revealAi.className = `reveal-value ${aiCorrect ? 'correct' : 'wrong'}`;
        revealUser.textContent = `${userChoice ? "Même" : "Différente"} ${userCorrect ? '✓' : '✗'}`;
        revealUser.className = `reveal-value ${userCorrect ? 'correct' : 'wrong'}`;
        challengeReveal.hidden = false;
        challengeReveal.scrollIntoView({ behavior: 'smooth', block: 'center' });
        challengeNextBtn.hidden = false;
        challengeNextBtn.textContent = 'NOUVEAU DÉFI';
    }

    // Drag and Drop
    [visUpload, nirUpload].forEach((area, index) => {
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.style.borderColor = 'var(--primary)';
        });
        area.addEventListener('dragleave', () => {
            area.style.borderColor = 'var(--glass-border)';
        });
        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.style.borderColor = 'var(--glass-border)';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                if (index === 0) {
                    visFile = file;
                    handlePreview(file, visPreview, visUpload.querySelector('.upload-placeholder'));
                } else {
                    nirFile = file;
                    handlePreview(file, nirPreview, nirUpload.querySelector('.upload-placeholder'));
                }
                checkReady();
            }
        });
    });
});
