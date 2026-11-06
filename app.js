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

function smoothPoseData(frames, radius = 2) {
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

function drawPose(landmarks) {
  return drawSelectedPose(selectPosePoints(landmarks));
}

function drawSelectedPose(points) {
  // The canvas starts hidden and must be made visible for both the first
  // completed AI analysis and subsequent timeline renders.
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
  if (visibleChain.length < 5) {
    throw new Error("肩、髖、膝、踝與腳尖未全部清楚入鏡");
  }
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
    poseContext.strokeStyle = "rgba(0, 0, 0, .72)";
    poseContext.lineWidth = Math.max(10, stageRect.width / 52);
    poseContext.shadowColor = "rgba(0, 0, 0, .35)";
    poseContext.shadowBlur = 5;
    poseContext.stroke();
    poseContext.shadowBlur = 0;
    poseContext.strokeStyle = "#ff3322";
    poseContext.lineWidth = Math.max(6, stageRect.width / 88);
    poseContext.stroke();
  });
  canvasChain.forEach((point, index) => {
    const { x, y } = point;
    poseContext.beginPath();
    poseContext.arc(x, y, Math.max(7, stageRect.width / 100), 0, Math.PI * 2);
    poseContext.fillStyle = chain[index]?.isManual ? "#f59e0b" : "#ff3322";
    poseContext.strokeStyle = "#ffffff";
    poseContext.lineWidth = 2;
    poseContext.fill();
    poseContext.stroke();
    poseContext.font = `700 ${Math.max(14, stageRect.width / 42)}px sans-serif`;
    poseContext.fillStyle = "#ffffff";
    poseContext.strokeStyle = "rgba(0, 0, 0, .8)";
    poseContext.lineWidth = 4;
    poseContext.strokeText(pointNames[index], x + 10, y - 10);
    poseContext.fillText(pointNames[index], x + 10, y - 10);
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

function drawPointChain(chain, label = "手動標記") {
  const stageRect = videoStage.getBoundingClientRect();
  const videoRect = getVideoContentRect();
  const dpr = window.devicePixelRatio || 1;
  poseCanvas.width = Math.max(1, Math.round(stageRect.width * dpr));
  poseCanvas.height = Math.max(1, Math.round(stageRect.height * dpr));
  poseCanvas.style.width = `${stageRect.width}px`;
  poseCanvas.style.height = `${stageRect.height}px`;
  poseContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  poseContext.clearRect(0, 0, stageRect.width, stageRect.height);
  const canvasChain = chain.map((point) => ({
    x: (videoRect.left - stageRect.left) + point.x * videoRect.width,
    y: (videoRect.top - stageRect.top) + point.y * videoRect.height
  }));
  poseContext.lineCap = "round";
  poseContext.lineJoin = "round";
  canvasChain.slice(0, -1).forEach((point, index) => {
    const next = canvasChain[index + 1];
    poseContext.beginPath();
    poseContext.moveTo(point.x, point.y);
    poseContext.lineTo(next.x, next.y);
    poseContext.strokeStyle = "rgba(0, 0, 0, .72)";
    poseContext.lineWidth = Math.max(10, stageRect.width / 52);
    poseContext.stroke();
    poseContext.strokeStyle = "#ff3322";
    poseContext.lineWidth = Math.max(6, stageRect.width / 88);
    poseContext.stroke();
  });
  canvasChain.forEach(({ x, y }, index) => {
    poseContext.beginPath();
    poseContext.arc(x, y, Math.max(7, stageRect.width / 100), 0, Math.PI * 2);
    poseContext.fillStyle = "#f59e0b";
    poseContext.strokeStyle = "#ffffff";
    poseContext.lineWidth = 2;
    poseContext.fill();
    poseContext.stroke();
    poseContext.font = `700 ${Math.max(14, stageRect.width / 42)}px sans-serif`;
    poseContext.fillStyle = "#ffffff";
    poseContext.strokeStyle = "rgba(0, 0, 0, .8)";
    poseContext.lineWidth = 4;
    poseContext.strokeText(["肩", "髖", "膝", "踝", "尖"][index], x + 10, y - 10);
    poseContext.fillText(["肩", "髖", "膝", "踝", "尖"][index], x + 10, y - 10);
  });
  poseCanvas.hidden = false;
  $("#poseFrameLabel").textContent = `${label}目前影格 · ${formatTime(videoPreview.currentTime)}`;
}

function showAngles(points, statusText) {
  const hip = angleAt(points[1], points[0], points[2]);
  const knee = angleAt(points[2], points[1], points[3]);
  const ankle = angleAt(points[3], points[2], points[4]);
  $("#hipAngle").textContent = `${hip.toFixed(1)}°`;
  $("#kneeAngle").textContent = `${knee.toFixed(1)}°`;
  $("#ankleAngle").textContent = `${ankle.toFixed(1)}°`;
  angleReadout.hidden = false;
  analysisStatus.className = "analysis-status success";
  analysisStatus.textContent = statusText;
}

function startManualPose() {
  manualPoseActive = true;
  manualPoints = [];
  // Freeze the video while the user is selecting points. Without pausing here,
  // a previously playing preview can continue moving underneath the clicks.
  videoPreview.pause();
  $("#playPause").innerHTML = "▶ <span>播放</span>";
  poseContext.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
  poseCanvas.style.pointerEvents = "auto";
  poseCanvas.style.cursor = "crosshair";
  manualPoseButton.classList.add("active");
  analysisStatus.className = "analysis-status";
  analysisStatus.textContent = "請依序點擊影片中的：肩 → 髖 → 膝 → 踝 → 腳尖。";
}

function recordManualPoint(event) {
  if (!manualPoseActive) return;
  // Keep a manual point selection from triggering the native <video>
  // play/pause action. The event must still bubble to this handler so the
  // selected point can be recorded.
  event.preventDefault();
  const videoRect = getVideoContentRect();
  const x = (event.clientX - videoRect.left) / videoRect.width;
  const y = (event.clientY - videoRect.top) / videoRect.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  manualPoints.push({ x, y, isManual: true });
  drawPointChain(manualPoints);
  if (manualPoints.length === 5) {
    manualPoseActive = false;
    poseCanvas.style.pointerEvents = "none";
    poseCanvas.style.cursor = "default";
    manualPoseButton.classList.remove("active");
    showAngles(manualPoints, "手動骨骼標記完成：已標記肩、髖、膝、踝與腳尖。");
  } else {
    const next = ["肩", "髖", "膝", "踝", "腳尖"][manualPoints.length];
    analysisStatus.textContent = `已標記 ${manualPoints.length}/5，下一個請點擊：${next}。`;
  }
}

videoStage.addEventListener("click", (event) => {
  if (!manualPoseActive || event.target === poseCanvas) return;
  event.preventDefault();
  recordManualPoint(event);
});

poseCanvas.addEventListener("click", (event) => {
  if (!manualPoseActive) return;
  event.preventDefault();
  event.stopPropagation();
  recordManualPoint(event);
});

videoPreview.addEventListener("click", (event) => {
  if (!manualPoseActive) return;
  // Prevent the video element's default click-to-play behaviour while
  // allowing the parent videoStage handler to register the point.
  event.preventDefault();
  videoPreview.pause();
});

videoPreview.addEventListener("play", () => {
  // Some browsers start playback before the click event reaches the parent.
  // Pause again as a guard so manual point selection never advances the frame.
  if (manualPoseActive) {
    videoPreview.pause();
    $("#playPause").innerHTML = "▶ <span>播放</span>";
  }
});

async function getPoseLandmarker() {
  if (poseLandmarker) return poseLandmarker;
  if (poseLoading) {
    while (poseLoading) await new Promise((resolve) => setTimeout(resolve, 100));
    return poseLandmarker;
  }
  poseLoading = true;
  analysisStatus.className = "analysis-status";
  analysisStatus.textContent = "正在載入 AI 骨骼模型，第一次可能需要幾秒鐘……";
  try {
    // Load MediaPipe from the project itself. This avoids relying on remote
    // CDNs (esm.sh, jsdelivr and Google Storage), which can be unavailable in
    // restricted browser/network environments.
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
  [frameRange, clipRange, $("#previousFrame"), $("#nextFrame"), $("#playPause"), $("#btnMarkS"), $("#takeoffButton"), $("#landingButton"), $("#btnMarkF"), manualPoseButton].forEach((element) => {
    if (element) element.disabled = locked;
  });
  autoPoseButton.disabled = locked;
  autoPoseButton.textContent = locked ? "分析中…" : "✦ AI 骨骼分析";
}

async function analyzeVideoPose() {
  if (!videoUrl || videoPreview.readyState < 2) {
    analysisStatus.className = "analysis-status error";
    analysisStatus.textContent = "請先匯入並載入影片，再執行骨骼分析。";
    return;
  }
  if (isAnalyzing) return;
  isAnalyzing = true;
  setAnalysisLock(true);
  try {
    const landmarker = await getPoseLandmarker();
    videoPreview.pause();
    const fps = Number(fpsSelect.value);
    const startFrame = Math.max(0, takeoffFrame ?? 0);
    const requestedEnd = landingFrame ?? Number(clipRange.value || totalFrames);
    const endFrame = Math.min(totalFrames, Math.max(startFrame, requestedEnd));
    if (endFrame <= startFrame) throw new Error("請先標記 T 腳尖離地與 L 首次落地，或設定有效的分析範圍");
    const rawPoseData = new Map();
    for (let frame = startFrame; frame <= endFrame; frame += 1) {
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
      if ((frame - startFrame) % 4 === 0 || frame === endFrame) {
        const progress = Math.round(((frame - startFrame) / (endFrame - startFrame)) * 100);
        analysisStatus.className = "analysis-status";
        analysisStatus.textContent = `AI 逐格分析中… ${progress}%（Frame ${frame}/${endFrame}）`;
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    poseData = smoothPoseData(rawPoseData);
    if (!poseData.size) throw new Error("在指定範圍內找不到清楚的人體姿態");
    poseAnalysisRange = { start: startFrame, end: endFrame };
    const firstFrame = poseData.keys().next().value;
    frameRange.value = String(firstFrame);
    videoPreview.currentTime = startFrame / fps;
    renderStoredPose(firstFrame);
    analysisStatus.className = "analysis-status success";
    analysisStatus.textContent = `AI 分析完成：已處理 ${poseData.size} 個影格，可拖曳時間軸檢視骨骼與角度。`;
  } catch (error) {
    console.error("Pose analysis failed:", error);
    analysisStatus.className = "analysis-status error";
    const detail = error?.message ? `（${error.message.slice(0, 90)}）` : "";
    analysisStatus.textContent = `AI 骨骼分析失敗${detail} 可改用「手動標記關節」完成紅點與角度分析。`;
  } finally {
    isAnalyzing = false;
    setAnalysisLock(false);
  }
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
  if (takeoffFrame === null || landingFrame === null) {
    analysisStatus.className = "analysis-status";
    analysisStatus.textContent = "請在逐格時間軸上標記 T 腳尖離地與 L 首次落地，結果會自動更新。";
    return;
  }
  if (landingFrame <= takeoffFrame) {
    analysisStatus.className = "analysis-status error";
    analysisStatus.textContent = "L 必須晚於 T，請把時間軸移到落地影格後再按下 L。";
    flightMetric.innerHTML = "— <i>s</i>";
    heightMetric.innerHTML = "— <i>cm</i>";
    resultFlight.innerHTML = "— <i>s</i>";
    resultHeight.innerHTML = "— <i>cm</i>";
    saveAttempt.disabled = true;
    return;
  }
  const seconds = (landingFrame - takeoffFrame) / Number(fpsSelect.value);
  const height = (9.81 * seconds * seconds * 100) / 8;
  const flight = `${seconds.toFixed(3)} <i>s</i>`;
  const jumpHeight = `${height.toFixed(1)} <i>cm</i>`;
  flightMetric.innerHTML = flight;
  heightMetric.innerHTML = jumpHeight;
  resultFlight.innerHTML = flight;
  resultHeight.innerHTML = jumpHeight;
  saveAttempt.disabled = false;
  analysisStatus.className = "analysis-status success";
  analysisStatus.textContent = `分析完成：T ${formatTime(takeoffFrame / Number(fpsSelect.value))} → L ${formatTime(landingFrame / Number(fpsSelect.value))}，結果已更新。`;
}

function validatePhaseMarkers() {
  const markers = [
    ["S", startFrame],
    ["T", takeoffFrame],
    ["L", landingFrame],
    ["F", finishFrame]
  ].filter(([, frame]) => frame !== null);
  for (let index = 1; index < markers.length; index += 1) {
    if (markers[index][1] <= markers[index - 1][1]) {
      analysisStatus.className = "analysis-status error";
      analysisStatus.textContent = `時間標記順序錯誤：應符合 S < T < L < F，目前 ${markers[index - 1][0]} 與 ${markers[index][0]} 需要重新確認。`;
      return false;
    }
  }
  return true;
}

function markFrame(type) {
  if (!videoUrl) return;
  if (type === "start") startFrame = getCurrentFrame();
  if (type === "takeoff") takeoffFrame = getCurrentFrame();
  if (type === "landing") landingFrame = getCurrentFrame();
  if (type === "finish") finishFrame = getCurrentFrame();
  updateReadouts();
  calculateResult();
  if (!validatePhaseMarkers()) return;
  if (takeoffFrame !== null && landingFrame !== null && landingFrame > takeoffFrame) {
    analyzeVideoPose();
  }
}

function enableVideoTools() {
  [frameRange, clipRange, $("#previousFrame"), $("#nextFrame"), $("#playPause"), $("#btnMarkS"), $("#takeoffButton"), $("#landingButton"), $("#btnMarkF"), autoPoseButton, manualPoseButton].forEach((el) => { el.disabled = false; });
  frameRange.max = String(totalFrames);
  clipRange.max = String(totalFrames);
  $("#durationLabel").textContent = formatTime(videoPreview.duration || 0);
  updateReadouts();
}

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
  videoPreview.hidden = false;
  emptyVideo.hidden = true;
  videoOverlay.hidden = false;
  $("#videoName").textContent = file.name;
  analysisStatus.className = "analysis-status";
  analysisStatus.textContent = "影片已載入，請拖曳逐格時間軸或使用上一格／下一格，再標記 T／L。";
  videoPreview.addEventListener("loadedmetadata", () => {
    totalFrames = Math.max(1, Math.round(videoPreview.duration * Number(fpsSelect.value)));
    enableVideoTools();
  }, { once: true });
}

function renderAttempts() {
  if (!attempts.length) {
    attemptList.innerHTML = '<div class="attempt-empty">完成一次標記後，試跳結果會顯示在這裡。</div>';
    return;
  }
  attemptList.innerHTML = attempts.map((attempt, index) => `
    <div class="attempt-row">
      <div><small>試跳 ${index + 1}</small><strong>${attempt.height} cm</strong></div>
      <span>${attempt.flight} s${index === attempts.length - 1 ? " · 最新" : ""}</span>
    </div>`).join("");
}

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
  if (!isAnalyzing && poseData.has(frame)) renderStoredPose(frame);
});
clipRange.addEventListener("input", () => { if (videoPreview.duration) videoPreview.currentTime = Number(clipRange.value) / Number(fpsSelect.value); });
fpsSelect.addEventListener("change", () => {
  if (videoUrl) {
    totalFrames = Math.max(1, Math.round((videoPreview.duration || 0) * Number(fpsSelect.value)));
    frameRange.max = String(totalFrames);
    clipRange.max = String(totalFrames);
    updateReadouts();
    calculateResult();
  }
});
videoPreview.addEventListener("timeupdate", () => {
  if (!videoPreview.seeking) {
    const currentFrame = Math.round(videoPreview.currentTime * Number(fpsSelect.value));
    frameRange.value = String(currentFrame);
    updateReadouts();
    if (!isAnalyzing && poseData.size) renderStoredPose(currentFrame);
  }
});
$("#previousFrame").addEventListener("click", () => { frameRange.value = String(Math.max(0, getCurrentFrame() - 1)); frameRange.dispatchEvent(new Event("input")); });
$("#nextFrame").addEventListener("click", () => { frameRange.value = String(Math.min(totalFrames, getCurrentFrame() + 1)); frameRange.dispatchEvent(new Event("input")); });
$("#playPause").addEventListener("click", () => {
  if (videoPreview.paused) { videoPreview.play(); $("#playPause").innerHTML = "Ⅱ <span>暫停</span>"; }
  else { videoPreview.pause(); $("#playPause").innerHTML = "▶ <span>播放</span>"; }
});
$("#takeoffButton").addEventListener("click", () => markFrame("takeoff"));
$("#landingButton").addEventListener("click", () => markFrame("landing"));
$("#btnMarkS").addEventListener("click", () => markFrame("start"));
$("#btnMarkF").addEventListener("click", () => markFrame("finish"));
autoPoseButton.addEventListener("click", analyzeVideoPose);
manualPoseButton.addEventListener("click", startManualPose);

saveAttempt.addEventListener("click", () => {
  if (takeoffFrame === null || landingFrame === null) return;
  const flight = (landingFrame - takeoffFrame) / Number(fpsSelect.value);
  const height = (9.81 * flight * flight * 100) / 8;
  if (attempts.length >= 3) attempts.shift();
  attempts.push({ flight: flight.toFixed(3), height: height.toFixed(1) });
  renderAttempts();
});

$("#copyResult").addEventListener("click", async () => {
  const text = `JumpLab 分析結果\nFlight Time: ${takeoffFrame !== null && landingFrame !== null ? ((landingFrame - takeoffFrame) / Number(fpsSelect.value)).toFixed(3) : "—"} s\nJump Height: ${resultHeight.textContent.trim()}`;
  try {
    await navigator.clipboard.writeText(text);
    $("#copyResult").textContent = "已複製 ✓";
    setTimeout(() => { $("#copyResult").textContent = "複製結果"; }, 1800);
  } catch { window.alert(text); }
});
$("#printReport").addEventListener("click", () => window.print());
// 暫時略過安全確認視窗，直接開放分析工具測試。
// 保留 modal 與原本的確認事件，之後若恢復正式流程只需改回 false。
safetyModal.classList.add("hidden");
setAppEnabled(true);
