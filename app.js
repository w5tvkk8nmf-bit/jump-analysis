// ============================================================================
// 1. DOM 元素選取與全域變數宣告
// ============================================================================

const $ = (selector) => document.querySelector(selector);

const safetyModal = $("#safetyModal");
const guideModal = $("#guideModal");
const safetyConfirm = $("#safetyConfirm");
const enterApp = $("#enterApp");
const videoInput = $("#videoInput");
const videoPreview = $("#videoPreview");
const emptyVideo = $("#emptyVideo");
const videoOverlay = $("#videoOverlay");
const frameRange = $("#frameRange");
const clipRange = $("#clipRange");
const fpsSelect = $("#fps");
const flightMetric = $("#flightMetric");
const heightMetric = $("#heightMetric");
const resultFlight = $("#resultFlight");
const resultHeight = $("#resultHeight");
const takeoffReadout = $("#takeoffReadout");
const landingReadout = $("#landingReadout");
const readoutS = $("#readoutS");
const readoutF = $("#readoutF");
const saveAttempt = $("#saveAttempt");
const attemptList = $("#attemptList");
const analysisStatus = $("#analysisStatus");
const poseCanvas = $("#poseCanvas");
const poseContext = poseCanvas.getContext("2d");
const angleReadout = $("#angleReadout");
const autoPoseButton = $("#autoPoseButton");
const manualPoseButton = $("#manualPoseButton");
const videoStage = $("#videoStage");

let videoUrl = "";
let totalFrames = 0;
let takeoffFrame = null;
let landingFrame = null;
let startFrame = null;
let finishFrame = null;
let attempts = [];
let poseLandmarker = null;
let poseLoading = false;
let lastPoseTimestamp = 0;
let poseData = new Map();
let poseAnalysisRange = null;
let isAnalyzing = false;
let manualPoseActive = false;
let manualPoints = [];
let kinematicsChartInstance = null;

// ============================================================================
// 2. 核心運算與自動分期演算法 (支援 S, T, L, F 全自動判斷)
// ============================================================================

const POSE_CONNECTIONS = [
  [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
  [23, 25], [25, 27], [24, 26], [26, 28], [27, 29], [28, 30],
  [27, 31], [28, 32], [11, 12], [23, 24]
];

function angleAt(vertex, pointA, pointB) {
  const a = Math.atan2(pointA.y - vertex.y, pointA.x - vertex.x);
  const b = Math.atan2(pointB.y - vertex.y, pointB.x - vertex.x);
  let degrees = Math.abs((a - b) * 180 / Math.PI);
  if (degrees > 180) degrees = 360 - degrees;
  return degrees;
}

function selectPosePoints(landmarks) {
  const visibility = (index) => landmarks[index]?.visibility ?? 0;
  const score = (indices) => indices.reduce((sum, index) => sum + visibility(index), 0);
  const left = score([11, 23, 25, 27, 31]) >= score([12, 24, 26, 28, 32]);
  return left
    ? { shoulder: landmarks[11], hip: landmarks[23], knee: landmarks[25], ankle: landmarks[27], toe: landmarks[31] }
    : { shoulder: landmarks[12], hip: landmarks[24], knee: landmarks[26], ankle: landmarks[28], toe: landmarks[32] };
}

function smoothPoseData(frames, radius = 3) {
  const result = new Map();
  [...frames.keys()].sort((a, b) => a - b).forEach((frame) => {
    const points = {};
    ["shoulder", "hip", "knee", "ankle", "toe"].forEach((name) => {
      const samples = [];
      for (let offset = -radius; offset <= radius; offset += 1) {
        const point = frames.get(frame + offset)?.[name];
        if (point) samples.push(point);
      }
      if (!samples.length) return;
      points[name] = {
        x: samples.reduce((sum, point) => sum + point.x, 0) / samples.length,
        y: samples.reduce((sum, point) => sum + point.y, 0) / samples.length,
        visibility: Math.min(...samples.map((point) => point.visibility ?? 1))
      };
    });
    if (Object.keys(points).length === 5) result.set(frame, points);
  });
  return result;
}

function autoDetectJumpPhases(poseDataMap) {
  const frames = [...poseDataMap.keys()].sort((a, b) => a - b);
  if (frames.length < 15) return null;

  let maxHipY = -Infinity;
  let autoS = null;

  frames.forEach((frame) => {
    const points = poseDataMap.get(frame);
    if (points.hip.y > maxHipY) {
      maxHipY = points.hip.y;
      autoS = frame;
    }
  });

  const baselineToeY = frames.slice(0, 10).reduce((sum, f) => sum + poseDataMap.get(f).toe.y, 0) / 10;
  const takeoffThreshold = baselineToeY - 0.015;

  let autoT = null;
  let autoL = null;
  let inAir = false;

  const framesAfterS = frames.filter(f => f >= autoS);
  for (let i = 0; i < framesAfterS.length; i++) {
    const frame = framesAfterS[i];
    const toeY = poseDataMap.get(frame).toe.y;

    if (!inAir && toeY < takeoffThreshold) {
      autoT = frame;
      inAir = true;
    } else if (inAir && toeY >= takeoffThreshold) {
      autoL = frame;
      break;
    }
  }

  let autoF = frames[frames.length - 1];
  if (autoL !== null) {
    let maxLandingHipY = -Infinity;
    frames.filter(f => f >= autoL && f <= autoL + 60).forEach((frame) => {
      const points = poseDataMap.get(frame);
      if (points && points.hip.y > maxLandingHipY) {
        maxLandingHipY = points.hip.y;
        autoF = frame;
      }
    });
  }

  return { autoS, autoT, autoL, autoF };
}

// ============================================================================
// 3. Canvas 繪圖與視覺化渲染
// ============================================================================

function getVideoContentRect() {
  const stageRect = videoStage.getBoundingClientRect();
  const videoRect = videoPreview.getBoundingClientRect();
  const sourceWidth = videoPreview.videoWidth || videoRect.width;
  const sourceHeight = videoPreview.videoHeight || videoRect.height;
  if (!sourceWidth || !sourceHeight) return videoRect;
  const sourceRatio = sourceWidth / sourceHeight;
  const boxRatio = videoRect.width / videoRect.height;
  if (sourceRatio > boxRatio) {
    const height = videoRect.width / sourceRatio;
    return { left: videoRect.left, top: videoRect.top + (videoRect.height - height) / 2, width: videoRect.width, height };
  }
  const width = videoRect.height * sourceRatio;
  return { left: videoRect.left + (videoRect.width - width) / 2, top: videoRect.top, width, height: videoRect.height };
}

function drawSelectedPose(points) {
  poseCanvas.hidden = false;
  const stage = $("#videoStage");
  const stageRect = stage.getBoundingClientRect();
  const videoRect = getVideoContentRect();
  const dpr = window.devicePixelRatio || 1;
  
  poseCanvas.width = Math.max(1, Math.round(stageRect.width * dpr));
  poseCanvas.height = Math.max(1, Math.round(stageRect.height * dpr));
  poseCanvas.style.width = `${stageRect.width}px`;
  poseCanvas.style.height = `${stageRect.height}px`;
  
  poseContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  poseContext.clearRect(0, 0, stageRect.width, stageRect.height);
  
  const pointNames = ["肩", "髖", "膝", "踝", "尖"];
  const chain = [points.shoulder, points.hip, points.knee, points.ankle, points.toe];
  const visibleChain = chain.filter((point) => point && (point.visibility ?? 1) >= 0.2);
  
  if (visibleChain.length < 5) return points;
  
  const toCanvasPoint = (point) => ({
    x: (videoRect.left - stageRect.left) + point.x * videoRect.width,
    y: (videoRect.top - stageRect.top) + point.y * videoRect.height
  });
  
  const canvasChain = chain.map(toCanvasPoint);
  
  poseContext.lineCap = "round";
  poseContext.lineJoin = "round";
  
  canvasChain.slice(0, -1).forEach((point, index) => {
    const next = canvasChain[index + 1];
    
    poseContext.beginPath();
    poseContext.moveTo(point.x, point.y);
    poseContext.lineTo(next.x, next.y);
    poseContext.strokeStyle = "rgba(0, 255, 255, 0.35)";
    poseContext.lineWidth = Math.max(8, stageRect.width / 80);
    poseContext.shadowColor = "#00ffff";
    poseContext.shadowBlur = 10;
    poseContext.stroke();
    
    poseContext.shadowBlur = 0;
    poseContext.beginPath();
    poseContext.moveTo(point.x, point.y);
    poseContext.lineTo(next.x, next.y);
    poseContext.strokeStyle = "rgba(10, 25, 30, 0.9)";
    poseContext.lineWidth = Math.max(4, stageRect.width / 130);
    poseContext.stroke();
    
    poseContext.beginPath();
    poseContext.moveTo(point.x, point.y);
    poseContext.lineTo(next.x, next.y);
    poseContext.strokeStyle = "#00ffff";
    poseContext.lineWidth = Math.max(1.5, stageRect.width / 250);
    poseContext.stroke();
  });
  
  canvasChain.forEach((point, index) => {
    const { x, y } = point;
    const baseRadius = Math.max(4, stageRect.width / 130);
    
    poseContext.beginPath();
    poseContext.arc(x, y, baseRadius + 2, 0, Math.PI * 2);
    poseContext.fillStyle = "rgba(10, 25, 30, 0.95)";
    poseContext.strokeStyle = "#00ffff";
    poseContext.lineWidth = 1.5;
    poseContext.shadowColor = "#00ffff";
    poseContext.shadowBlur = 6;
    poseContext.fill();
    poseContext.stroke();
    
    poseContext.shadowBlur = 0;
    poseContext.beginPath();
    poseContext.arc(x, y, Math.max(1.5, baseRadius - 1.5), 0, Math.PI * 2);
    poseContext.fillStyle = chain[index]?.isManual ? "#f59e0b" : "#ffffff";
    poseContext.fill();
    
    const fontSize = Math.max(12, stageRect.width / 55);
    poseContext.font = `600 ${fontSize}px sans-serif`;
    const textWidth = poseContext.measureText(pointNames[index]).width;
    
    poseContext.fillStyle = "rgba(0, 15, 20, 0.75)";
    poseContext.fillRect(x + 10, y - fontSize - 4, textWidth + 10, fontSize + 8);
    
    poseContext.fillStyle = "#00ffff";
    poseContext.fillRect(x + 10, y - fontSize - 4, 3, fontSize + 8);
    
    poseContext.fillStyle = "#00ffff";
    poseContext.fillText(pointNames[index], x + 16, y - 2);
  });
  
  return points;
}

function renderStoredPose(frame) {
  let renderedFrame = frame;
  let points = poseData.get(frame);
  if (!points && poseAnalysisRange && frame >= poseAnalysisRange.start && frame <= poseAnalysisRange.end && poseData.size) {
    renderedFrame = [...poseData.keys()].reduce((nearest, candidate) => (
      Math.abs(candidate - frame) < Math.abs(nearest - frame) ? candidate : nearest
    ));
    points = poseData.get(renderedFrame);
  }
  if (!points) {
    clearPoseOverlay();
    return;
  }
  drawSelectedPose(points);
  showAngles([points.shoulder, points.hip, points.knee, points.ankle, points.toe], `AI 骨骼標記 · Frame ${renderedFrame}`);
}

function clearPoseOverlay() {
  poseContext.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
  poseCanvas.hidden = true;
  angleReadout.hidden = true;
}

function showAngles(points, statusText) {
  const hip = angleAt(points[1], points[0], points[2]);
  const knee = angleAt(points[2], points[1], points[3]);
  const ankle = angleAt(points[3], points[2], points[4]);
  $("#hipAngle").textContent = `${(180 - hip).toFixed(1)}°`;
  $("#kneeAngle").textContent = `${(180 - knee).toFixed(1)}°`;
  $("#ankleAngle").textContent = `${(90 - angleAt(points[4], points[3], points[2])).toFixed(1)}°`;
  angleReadout.hidden = false;
  analysisStatus.className = "analysis-status success";
  analysisStatus.textContent = statusText;
}

// ============================================================================
// 4. MediaPipe AI 模型載入與影片處理
// ============================================================================

async function getPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;
  if (poseLoading) {
    while (poseLoading) await new Promise((resolve) => setTimeout(resolve, 100));
    return poseLandmarker;
  }
  poseLoading = true;
  analysisStatus.className = "analysis-status";
  analysisStatus.textContent = "正在載入 AI 骨骼模型……";
  try {
    const { FilesetResolver, PoseLandmarker } = await import("./vendor/mediapipe/vision_bundle.mjs");
    const vision = await FilesetResolver.forVisionTasks(new URL("./vendor/mediapipe/wasm", window.location.href).href);
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: new URL("./models/pose_landmarker_lite.task", window.location.href).href
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: .3,
      minPosePresenceConfidence: .3,
      minTrackingConfidence: .3
    });
    return poseLandmarker;
  } finally {
    poseLoading = false;
  }
}

function waitForSeek(video, targetTime) {
  if (Math.abs(video.currentTime - targetTime) < 0.0005 && video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    let settled = false;
    const cleanup = () => {
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      clearTimeout(timeout);
    };
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      cleanup();
      callback();
    };
    const onSeeked = () => finish(resolve);
    const onError = () => finish(() => reject(new Error("影片影格載入失敗")));
    const timeout = setTimeout(() => finish(() => reject(new Error("等待影片影格逾時"))), 4000);
    
    video.addEventListener("seeked", onSeeked, { once: true });
    video.addEventListener("error", onError, { once: true });
    video.currentTime = targetTime;
  });
}

function setAnalysisLock(locked) {
  [frameRange, clipRange, $("#previousFrame"), $("#nextFrame"), $("#playPause"), autoPoseButton].forEach((element) => {
    if (element) element.disabled = locked;
  });
  autoPoseButton.textContent = locked ? "分析中…" : "✦ AI 骨骼分析";
}

async function analyzeVideoPose() {
  if (!videoUrl || videoPreview.readyState < 2) return;
  if (isAnalyzing) return;
  isAnalyzing = true;
  setAnalysisLock(true);
  try {
    const landmarker = await getPoseLandmarker();
    videoPreview.pause();
    const fps = Number(fpsSelect.value);
    
    const analysisStart = 0;
    const analysisEnd = totalFrames;
    
    const rawPoseData = new Map();
    for (let frame = analysisStart; frame <= analysisEnd; frame += 1) {
      const time = frame / fps;
      await waitForSeek(videoPreview, time);
      const timestamp = Math.max(Math.round(time * 1000), lastPoseTimestamp + 1);
      lastPoseTimestamp = timestamp;
      const result = landmarker.detectForVideo(videoPreview, timestamp);
      const landmarks = result?.landmarks?.[0];
      if (landmarks) {
        const points = selectPosePoints(landmarks);
        const visible = [points.shoulder, points.hip, points.knee, points.ankle, points.toe]
          .every((point) => point && (point.visibility ?? 1) >= .2);
        if (visible) rawPoseData.set(frame, points);
      }
      
      if (frame % 4 === 0 || frame === analysisEnd) {
        const progress = Math.round((frame / analysisEnd) * 100);
        analysisStatus.className = "analysis-status";
        analysisStatus.textContent = `AI 自動分析中… ${progress}%（Frame ${frame}/${analysisEnd}）`;
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    
    poseData = smoothPoseData(rawPoseData, 3);
    if (!poseData.size) throw new Error("在影片中找不到清楚的人體姿態");

    const phases = autoDetectJumpPhases(poseData);
    if (phases && phases.autoT && phases.autoL) {
      startFrame = phases.autoS;
      takeoffFrame = phases.autoT;
      landingFrame = phases.autoL;
      finishFrame = phases.autoF;
      updateReadouts();
      calculateResult();
    }

    poseAnalysisRange = { start: analysisStart, end: analysisEnd };
    
    frameRange.value = "0";
    videoPreview.currentTime = 0;
    renderStoredPose(0);
    
    analysisStatus.className = "analysis-status success";
    analysisStatus.textContent = `AI 分析完成：已自動標記 S / T / L / F，並產出運動學曲線圖。`;
    
    renderKinematicsChart();

  } catch (error) {
    console.error("Pose analysis failed:", error);
    analysisStatus.className = "analysis-status error";
    analysisStatus.textContent = `AI 自動分析失敗，請確認影片包含完整全身入鏡。`;
  } finally {
    isAnalyzing = false;
    setAnalysisLock(false);
  }
}

// ============================================================================
// 5. 圖表渲染與 UI 互動邏輯
// ============================================================================

function calculateFlexionAngles(points) {
  const hipAngle = angleAt(points.hip, points.shoulder, points.knee);
  const kneeAngle = angleAt(points.knee, points.hip, points.ankle);
  const ankleAngle = angleAt(points.ankle, points.knee, points.toe);
  return {
    hip: 180 - hipAngle,
    knee: 180 - kneeAngle,
    ankle: 90 - ankleAngle
  };
}

function renderKinematicsChart() {
  if (!poseData.size || !poseAnalysisRange) return;
  
  const chartContainer = document.getElementById("chartSection");
  if (chartContainer) chartContainer.style.display = "block";
  
  const canvasEl = document.getElementById("kinematicsChart");
  if (!canvasEl) return;
  
  const ctx = canvasEl.getContext("2d");
  const fps = Number(fpsSelect.value);
  const labels = [];
  const dataHip = [];
  const dataKnee = [];
  const dataAnkle = [];
  
  for (let frame = poseAnalysisRange.start; frame <= poseAnalysisRange.end; frame++) {
    const points = poseData.get(frame);
    if (points) {
      const angles = calculateFlexionAngles(points);
      labels.push((frame / fps).toFixed(3));
      dataHip.push(angles.hip);
      dataKnee.push(angles.knee);
      dataAnkle.push(angles.ankle);
    }
  }

  if (kinematicsChartInstance) kinematicsChartInstance.destroy();

  kinematicsChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: '髖關節屈曲 (°)', data: dataHip, borderColor: '#2c684f', borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: '膝關節屈曲 (°)', data: dataKnee, borderColor: '#00ffff', borderWidth: 2, pointRadius: 0, tension: 0.3 },
        { label: '踝關節屈曲 (°)', data: dataAnkle, borderColor: '#f59e0b', borderWidth: 2, pointRadius: 0, tension: 0.3 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { title: { display: true, text: '時間 (秒)' } },
        y: { title: { display: true, text: '角度 (°)' } }
      },
      onClick: (event, elements, chart) => {
        if (elements.length > 0) {
          const dataIndex = elements[0].index;
          const targetTime = parseFloat(chart.data.labels[dataIndex]);
          videoPreview.currentTime = targetTime;
          frameRange.value = Math.round(targetTime * fps);
          document.querySelector("#analyzer").scrollIntoView({ behavior: "smooth" });
        }
      }
    }
  });
}

const formatTime = (seconds) => {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(3).padStart(6, "0")}`;
};

const getCurrentFrame = () => Number(frameRange.value || 0);

function setAppEnabled(enabled) {
  [videoInput, $("#testType"), fpsSelect, $("#uploadButton"), $("#heroUpload")].forEach((element) => {
    if (element) element.disabled = !enabled;
  });
}

function updateReadouts() {
  const fps = Number(fpsSelect.value);
  const current = getCurrentFrame();
  readoutS.textContent = `S ${startFrame === null ? "—" : formatTime(startFrame / fps)}`;
  takeoffReadout.textContent = `T ${takeoffFrame === null ? "—" : formatTime(takeoffFrame / fps)}`;
  landingReadout.textContent = `L ${landingFrame === null ? "—" : formatTime(landingFrame / fps)}`;
  readoutF.textContent = `F ${finishFrame === null ? "—" : formatTime(finishFrame / fps)}`;
  $("#frameLabel").textContent = `Frame ${current} / ${totalFrames || 1}`;
  $("#videoTime").textContent = formatTime(videoPreview.currentTime || current / fps);
}

function calculateResult() {
  if (takeoffFrame === null || landingFrame === null) return;
  if (landingFrame <= takeoffFrame) return;
  
  const seconds = (landingFrame - takeoffFrame) / Number(fpsSelect.value);
  const height = (9.81 * seconds * seconds * 100) / 8;
  const flight = `${seconds.toFixed(3)} <i>s</i>`;
  const jumpHeight = `${height.toFixed(1)} <i>cm</i>`;
  
  flightMetric.innerHTML = flight;
  heightMetric.innerHTML = jumpHeight;
  resultFlight.innerHTML = flight;
  resultHeight.innerHTML = jumpHeight;
  saveAttempt.disabled = false;
}

// 修正後的影片載入與事件觸發
function loadVideo(file) {
  if (!file) return;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  
  videoUrl = URL.createObjectURL(file);
  poseData = new Map();
  poseAnalysisRange = null;
  takeoffFrame = null;
  landingFrame = null;
  startFrame = null;
  finishFrame = null;
  
  clearPoseOverlay();
  videoPreview.src = videoUrl;
  videoPreview.load();
  videoPreview.hidden = false;
  emptyVideo.hidden = true;
  videoOverlay.hidden = false;
  $("#videoName").textContent = file.name;
  analysisStatus.className = "analysis-status";
  analysisStatus.textContent = "影片已載入，正在讀取中繼資料...";
  
  videoPreview.addEventListener("loadedmetadata", async () => {
    totalFrames = Math.max(1, Math.round(videoPreview.duration * Number(fpsSelect.value)));
    enableVideoTools();
    analysisStatus.textContent = "影片已就緒，AI 正在自動掃描全片...";
    await analyzeVideoPose();
  }, { once: true });

  videoPreview.addEventListener("error", (e) => {
    console.error("HTML5 Video 錯誤:", videoPreview.error);
    analysisStatus.className = "analysis-status error";
    analysisStatus.textContent = "瀏覽器無法解碼此影片格式，建議轉換為標準 MP4 (H.264) 後重試。";
  }, { once: true });
}

function enableVideoTools() {
  [frameRange, clipRange, $("#previousFrame"), $("#nextFrame"), $("#playPause"), autoPoseButton].forEach((el) => { el.disabled = false; });
  frameRange.max = String(totalFrames);
  clipRange.max = String(totalFrames);
  $("#durationLabel").textContent = formatTime(videoPreview.duration || 0);
  updateReadouts();
}

// ============================================================================
// 6. 事件監聽與綁定區
// ============================================================================

safetyConfirm.addEventListener("change", () => { enterApp.disabled = !safetyConfirm.checked; });
enterApp.addEventListener("click", () => { safetyModal.classList.add("hidden"); setAppEnabled(true); });
$("#openGuide").addEventListener("click", () => guideModal.classList.remove("hidden"));
$("#closeGuide").addEventListener("click", () => guideModal.classList.add("hidden"));
guideModal.addEventListener("click", (event) => { if (event.target === guideModal) guideModal.classList.add("hidden"); });

$("#uploadButton").addEventListener("click", () => videoInput.click());
$("#heroUpload").addEventListener("click", () => { document.querySelector("#analyzer").scrollIntoView({ behavior: "smooth" }); videoInput.click(); });
videoInput.addEventListener("change", (event) => loadVideo(event.target.files[0]));

frameRange.addEventListener("input", () => {
  const frame = getCurrentFrame();
  if (videoPreview.duration) videoPreview.currentTime = frame / Number(fpsSelect.value);
  updateReadouts();
  renderStoredPose(frame);
});

videoPreview.addEventListener("timeupdate", () => {
  if (!videoPreview.seeking) {
    const currentFrame = Math.round(videoPreview.currentTime * Number(fpsSelect.value));
    frameRange.value = String(currentFrame);
    updateReadouts();
    if (poseData.size) renderStoredPose(currentFrame);
  }
});

$("#previousFrame").addEventListener("click", () => { frameRange.value = String(Math.max(0, getCurrentFrame() - 1)); frameRange.dispatchEvent(new Event("input")); });
$("#nextFrame").addEventListener("click", () => { frameRange.value = String(Math.min(totalFrames, getCurrentFrame() + 1)); frameRange.dispatchEvent(new Event("input")); });
$("#playPause").addEventListener("click", () => {
  if (videoPreview.paused) { videoPreview.play(); $("#playPause").innerHTML = "Ⅱ <span>暫停</span>"; }
  else { videoPreview.pause(); $("#playPause").innerHTML = "▶ <span>播放</span>"; }
});

autoPoseButton.addEventListener("click", analyzeVideoPose);

safetyModal.classList.add("hidden");
setAppEnabled(true);
