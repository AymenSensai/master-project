document.addEventListener('DOMContentLoaded', () => {
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

    const modeVerify = document.getElementById('mode-verify');
    const modeRecognize = document.getElementById('mode-recognize');
    const identityList = document.getElementById('identity-list');
    const faceCanvas = document.getElementById('face-canvas');

    let visFile = null;
    let nirFile = null;
    let currentMode = 'verify'; // 'verify' or 'recognize'

    // Mode Switching
    modeVerify.addEventListener('click', () => switchMode('verify'));
    modeRecognize.addEventListener('click', () => switchMode('recognize'));

    function switchMode(mode) {
        currentMode = mode;
        document.body.classList.toggle('recognition-mode', mode === 'recognize');
        modeVerify.classList.toggle('active', mode === 'verify');
        modeRecognize.classList.toggle('active', mode === 'recognize');
        
        // Update Action Button Text
        btnText.textContent = mode === 'verify' ? 'VERIFY IDENTITY' : 'RECOGNIZE FACE';
        
        // Reset results and check readiness
        resultContainer.hidden = true;
        checkReady();
        
        if (mode === 'recognize') {
            loadIdentities();
        }
    }

    async function loadIdentities() {
        try {
            const response = await fetch('/api/identities');
            const data = await response.json();
            
            identityList.innerHTML = '';
            if (data.identities.length === 0) {
                identityList.innerHTML = '<span class="identity-badge">Gallery is empty</span>';
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

        setLoading(true, 'ANALYZING...');
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
            if (data.error) alert('Error: ' + data.error);
            else displayResult(data);
        } catch (error) {
            console.error('Fetch error:', error);
            alert('An error occurred during verification.');
        } finally {
            setLoading(false, 'VERIFY IDENTITY');
        }
    }

    async function performRecognition() {
        if (!nirFile) return;

        setLoading(true, 'RECOGNIZING...');
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
                alert('Error: ' + data.error);
            } else if (data.message === 'Gallery is empty') {
                alert('Gallery is empty. Please add images to the gallery/ folder.');
            } else {
                displayRecognitionResult(data);
                if (data.face_box) {
                    drawFaceHighlight(data.face_box);
                }
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('An error occurred during recognition.');
        } finally {
            setLoading(false, 'RECOGNIZE FACE');
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
        ctx.strokeStyle = '#00f2fe';
        ctx.lineWidth = 3;
        ctx.shadowBlur = 15;
        ctx.shadowColor = '#00f2fe';
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
            resultStatus.textContent = 'IDENTITY MATCHED';
            resultMessage.textContent = `High confidence cross-spectral match. The NIR probe matches the VIS reference identity.`;
        } else {
            resultContainer.classList.add('mismatch');
            resultStatus.textContent = 'IDENTITY MISMATCH';
            resultMessage.textContent = `Low similarity detected. These appear to be different identities.`;
        }

        updateSimilarityBar(simPercent);
    }

    function displayRecognitionResult(data) {
        resultContainer.hidden = false;
        resultContainer.classList.remove('match', 'mismatch');
        
        const simPercent = (data.similarity * 100).toFixed(2);
        
        if (data.matched) {
            resultContainer.classList.add('match');
            resultStatus.textContent = `RECOGNIZED: ${data.identity}`;
            resultMessage.textContent = `Highest similarity match found in the gallery for ${data.identity} with ${simPercent}% confidence.`;
        } else {
            resultContainer.classList.add('mismatch');
            resultStatus.textContent = 'UNKNOWN IDENTITY';
            resultMessage.textContent = `No reliable match found in the gallery. Best match was ${data.identity} but with low confidence (${simPercent}%).`;
        }

        updateSimilarityBar(simPercent);
    }

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
