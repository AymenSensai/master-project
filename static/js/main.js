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

    const modeDashboard = document.getElementById('mode-dashboard');
    const dashboardSection = document.getElementById('dashboard-section');
    const statTotal = document.getElementById('stat-total');
    const statPrecision = document.getElementById('stat-precision');
    const statAccuracy = document.getElementById('stat-accuracy');
    const distMale = document.getElementById('dist-male');
    const distFemale = document.getElementById('dist-female');
    const distAsian = document.getElementById('dist-asian');
    const distOther = document.getElementById('dist-other');
    const barMale = document.getElementById('bar-male');
    const barFemale = document.getElementById('bar-female');
    const barAsian = document.getElementById('bar-asian');
    const barOther = document.getElementById('bar-other');

    const modeVerify = document.getElementById('mode-verify');
    const modeRecognize = document.getElementById('mode-recognize');
    const modeLive = document.getElementById('mode-live');
    const identityList = document.getElementById('identity-list');
    const faceCanvas = document.getElementById('face-canvas');

    const webcamCard = document.getElementById('webcam-card');
    const webcamVideo = document.getElementById('webcam-video');
    const webcamCanvas = document.getElementById('webcam-canvas');
    const webcamStatus = document.getElementById('webcam-status');
    const insightsPanel = document.getElementById('insights-panel');
    const xaiOriginal = document.getElementById('xai-original');
    const xaiHeatmap = document.getElementById('xai-heatmap');
    const closeInsights = document.getElementById('close-insights');

    let visFile = null;
    let nirFile = null;
    let currentMode = 'verify'; // 'verify', 'recognize', or 'live'
    let webcamStream = null;
    let recognitionInterval = null;

    // Mode Switching
    modeVerify.addEventListener('click', () => switchMode('verify'));
    modeRecognize.addEventListener('click', () => switchMode('recognize'));
    modeLive.addEventListener('click', () => switchMode('live'));
    modeDashboard.addEventListener('click', () => switchMode('dashboard'));

    async function switchMode(mode) {
        currentMode = mode;
        document.body.className = `${mode}-mode`;
        
        modeVerify.classList.toggle('active', mode === 'verify');
        modeRecognize.classList.toggle('active', mode === 'recognize');
        modeLive.classList.toggle('active', mode === 'live');
        modeDashboard.classList.toggle('active', mode === 'dashboard');
        
        // Update Action Button Text
        btnText.textContent = (mode === 'live' || mode === 'dashboard') ? '' : (mode === 'verify' ? 'VÉRIFIER L\'IDENTITÉ' : 'RECONNAÎTRE LE VISAGE');
        
        // Reset results and check readiness
        resultContainer.hidden = true;
        webcamCard.hidden = mode !== 'live';
        dashboardSection.hidden = mode !== 'dashboard';
        
        const comparisonGrid = document.querySelector('.comparison-grid');
        const actionSection = document.querySelector('.action-section');
        const galleryPeek = document.querySelector('.gallery-peek');
        const resultsContainer = document.getElementById('results-container');
        
        if (comparisonGrid) {
            comparisonGrid.style.display = (mode === 'dashboard' || mode === 'live') ? 'none' : '';
        }
        if (actionSection) {
            actionSection.style.display = (mode === 'dashboard' || mode === 'live') ? 'none' : '';
        }
        if (galleryPeek) {
            galleryPeek.style.display = mode === 'dashboard' ? 'none' : '';
        }
        if (resultsContainer && mode === 'dashboard') {
            resultsContainer.hidden = true;
        }
        
        if (mode === 'dashboard') {
            loadAnalytics();
        }
        
        checkReady();
        
        // Handle Webcam Lifecycle
        if (mode === 'live') {
            await startWebcam();
        } else {
            stopWebcam();
        }

        if (mode === 'recognize' || mode === 'live') {
            loadIdentities();
        }
    }

    async function loadAnalytics() {
        try {
            const response = await fetch('/api/analytics');
            const data = await response.json();
            
            // Update stats with animation
            animateValue(statTotal, 0, data.total_identities, 1000);
            
            if (statPrecision && data.model_precision) statPrecision.textContent = data.model_precision;
            if (statAccuracy && data.cross_spectral_accuracy) statAccuracy.textContent = data.cross_spectral_accuracy;
            
            // Update demographics
            const total = data.total_identities;
            const male = data.gender_distribution.Male;
            const female = data.gender_distribution.Female;
            const asian = data.ethnicity_distribution.Asian;
            const other = data.ethnicity_distribution.Other;
            
            distMale.textContent = male;
            distFemale.textContent = female;
            distAsian.textContent = asian;
            distOther.textContent = other;
            
            barMale.style.width = `${(male / total) * 100}%`;
            barFemale.style.width = `${(female / total) * 100}%`;
            barAsian.style.width = `${(asian / total) * 100}%`;
            barOther.style.width = `${(other / total) * 100}%`;
            
        } catch (error) {
            console.error('Error loading analytics:', error);
        }
    }

    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }

    let frameCount = 0;

    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 1280, height: 720, facingMode: 'user' } 
            });
            webcamVideo.srcObject = webcamStream;
            webcamStatus.textContent = 'Caméra active - Démarrage de la boucle...';
            
            // Wait for video metadata
            webcamVideo.onloadedmetadata = () => {
                recognitionInterval = setInterval(() => {
                    performLiveRecognition();
                }, 600);
            };
            
        } catch (err) {
            console.error('Webcam Error:', err);
            webcamStatus.textContent = 'Erreur: Accès refusé à la caméra';
            alert('Impossible d\'accéder à la webcam. Veuillez vérifier vos autorisations.');
        }
    }

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        if (recognitionInterval) {
            clearInterval(recognitionInterval);
            recognitionInterval = null;
        }
        const ctx = webcamCanvas.getContext('2d');
        ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
        webcamStatus.textContent = 'Caméra éteinte';
    }

    const webcamLiveStats = document.getElementById('webcam-live-stats');
    const liveIdentity = document.getElementById('live-identity');
    const liveSimilarity = document.getElementById('live-similarity');

    async function performLiveRecognition() {
        if (!webcamStream || currentMode !== 'live' || webcamVideo.videoWidth === 0) return;

        frameCount++;
        webcamStatus.textContent = `Analyse... [Image ${frameCount}]`;
        webcamLiveStats.hidden = false;

        // Capture current frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = webcamVideo.videoWidth;
        tempCanvas.height = webcamVideo.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(webcamVideo, 0, 0);
        
        // Convert to blob
        const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.7));
        if (!blob) return;

        const formData = new FormData();
        formData.append('image', blob, 'webcam.jpg');

        try {
            const response = await fetch('/api/recognize', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                webcamStatus.textContent = `Erreur API: ${data.error}`;
                return;
            }

            updateLiveStats(data);

            if (data.face_box) {
                drawLiveOverlay(data);
                webcamStatus.textContent = data.matched ? `Correspondance: ${data.identity}` : 'Visage détecté - Recherche...';
            } else {
                const ctx = webcamCanvas.getContext('2d');
                ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
                webcamStatus.textContent = 'Aucun visage détecté';
            }
        } catch (error) {
            console.error('Live Recognition Error:', error);
            webcamStatus.textContent = 'Erreur de connexion';
        }
    }

    function updateLiveStats(data) {
        if (data.identity) {
            liveIdentity.textContent = data.identity;
            liveSimilarity.textContent = `${(data.similarity * 100).toFixed(1)}%`;
            
            // Highlight color based on match
            liveIdentity.style.color = data.matched ? '#059669' : '#2563eb';
        } else {
            liveIdentity.textContent = '-';
            liveSimilarity.textContent = '0%';
        }
    }

    function drawLiveOverlay(data) {
        const ctx = webcamCanvas.getContext('2d');
        
        // Match canvas size to display size
        const displayWidth = webcamVideo.clientWidth;
        const displayHeight = webcamVideo.clientHeight;
        
        if (webcamCanvas.width !== displayWidth || webcamCanvas.height !== displayHeight) {
            webcamCanvas.width = displayWidth;
            webcamCanvas.height = displayHeight;
        }
        
        ctx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
        
        if (!data.face_box) return;

        const box = data.face_box;
        const scaleX = webcamCanvas.width / box.img_w;
        const scaleY = webcamCanvas.height / box.img_h;

        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        const isMatch = data.matched;
        const color = isMatch ? '#059669' : '#2563eb'; 

        // Draw Bounding Box
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.lineJoin = "round";
        ctx.strokeRect(x, y, w, h);

        // Label Background
        const label = isMatch ? `${data.identity} (${(data.similarity * 100).toFixed(0)}%)` : 'DÉTECTION...';
        ctx.font = '700 14px Inter';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        
        ctx.fillStyle = color;
        ctx.fillRect(x - 2, y - 35, textWidth + 24, 35);
        
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 10, y - 12);
        
        // Pulse Effect for match
        if (isMatch) {
            ctx.shadowBlur = 15;
            ctx.shadowColor = color;
            ctx.strokeRect(x, y, w, h);
            ctx.shadowBlur = 0;
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

    // Handle Upload Clicks
    visUpload.addEventListener('click', () => visInput.click());
    nirUpload.addEventListener('click', () => nirInput.click());

    // Handle File Selection
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

    // Action Button Handler
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
            const response = await fetch('/api/compare', {
                method: 'POST',
                body: formData
            });
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
            const response = await fetch('/api/recognize', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                alert('Erreur: ' + data.error);
            } else if (data.message === 'Gallery is empty') {
                alert('La galerie est vide. Veuillez ajouter des images dans le dossier gallery/.');
            } else {
                displayRecognitionResult(data);
                if (data.face_box) {
                    drawFaceHighlight(data.face_box);
                }
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
        const img = nirPreview;
        
        // Match canvas size to displayed image size
        faceCanvas.width = img.clientWidth;
        faceCanvas.height = img.clientHeight;
        faceCanvas.hidden = false;

        const scaleX = faceCanvas.width / box.img_w;
        const scaleY = faceCanvas.height / box.img_h;

        const centerX = (box.x + box.w / 2) * scaleX;
        const centerY = (box.y + box.h / 2) * scaleY;
        const radius = (Math.max(box.w, box.h) / 2) * Math.min(scaleX, scaleY) * 1.2;

        ctx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
        
        // Draw Glow/Circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 3;
        ctx.shadowBlur = 15;
        ctx.shadowColor = '#2563eb';
        ctx.stroke();

        // Optional: pulsed effect if we wanted to animate here
    }

    function setLoading(isLoading, text) {
        verifyBtn.disabled = isLoading;
        btnLoader.hidden = !isLoading;
        btnText.textContent = text;
        if (isLoading) faceCanvas.hidden = true;
    }

    function displayResult(data) {
        resultContainer.hidden = false;
        resultContainer.classList.remove('match', 'mismatch');
        
        const simPercent = (data.similarity * 100).toFixed(2);
        
        if (data.match) {
            resultContainer.classList.add('match');
            resultStatus.textContent = 'IDENTITÉ CORRESPONDANTE';
            resultMessage.textContent = `Correspondance Spectrale de haute confiance. La sonde NIR correspond à l'identité de référence VIS.`;
            
            if (data.heatmap) {
                addXAIButton(data.heatmap);
            }
        } else {
            resultContainer.classList.add('mismatch');
            resultStatus.textContent = 'IDENTITÉ NON CORRESPONDANTE';
            resultMessage.textContent = `Faible similarité détectée. Il semble s'agir de personnes différentes.`;
            removeXAIButton();
        }

        updateSimilarityBar(simPercent);
    }


    function displayRecognitionResult(data) {
        resultContainer.hidden = false;
        resultContainer.classList.remove('match', 'mismatch');
        
        const simPercent = (data.similarity * 100).toFixed(2);
        
        if (data.matched) {
            resultContainer.classList.add('match');
            resultStatus.textContent = `RECONNU : ${data.identity}`;
            resultMessage.textContent = `La meilleure correspondance trouvée dans la galerie pour ${data.identity} avec ${simPercent}% de confiance.`;
            
            if (data.heatmap) {
                addXAIButton(data.heatmap);
            }
        } else {
            resultContainer.classList.add('mismatch');
            resultStatus.textContent = 'IDENTITÉ INCONNUE';
            resultMessage.textContent = `Aucune correspondance fiable trouvée dans la galerie. La meilleure était ${data.identity} avec une faible confiance (${simPercent}%).`;
            removeXAIButton();
        }

        updateSimilarityBar(simPercent);
    }

    function addXAIButton(heatmapB64) {
        removeXAIButton(); // Clear existing
        const btn = document.createElement('button');
        btn.id = 'btn-show-xai';
        btn.className = 'btn-insight';
        btn.innerHTML = '🔍 POURQUOI CELA A-T-IL CORRESPOND ? (APERÇUS IA)';
        btn.addEventListener('click', () => {
            xaiOriginal.src = nirPreview.src;
            xaiHeatmap.src = `data:image/jpeg;base64,${heatmapB64}`;
            insightsPanel.hidden = false;
            insightsPanel.scrollIntoView({ behavior: 'smooth' });
        });
        resultContainer.appendChild(btn);
    }

    function removeXAIButton() {
        const existing = document.getElementById('btn-show-xai');
        if (existing) existing.remove();
        insightsPanel.hidden = true;
    }

    closeInsights.addEventListener('click', () => {
        insightsPanel.hidden = true;
    });


    function updateSimilarityBar(percent) {
        similarityValue.textContent = `${percent}%`;
        const displayWidth = Math.max(0, Math.min(100, percent));
        similarityFill.style.width = '0%';
        setTimeout(() => {
            similarityFill.style.width = `${displayWidth}%`;
        }, 50);
    }

    // Drag and Drop Support
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
