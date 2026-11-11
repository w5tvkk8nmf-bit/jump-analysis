// ============================================================================
// JumpLab 核心控制與影片載入腳本
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
const analysisStatus = $("#analysisStatus");
const poseCanvas = $("#poseCanvas");
const poseContext = poseCanvas.getContext("2d");
const angleReadout = $("#angleReadout");
const autoPoseButton = $("#autoPoseButton");
const videoStage = $("#videoStage");

let videoUrl = "";
let totalFrames = 0;
let takeoffFrame = null;
let landingFrame = null;
let startFrame = null;
let finishFrame = null;
let poseLandmarker = null;
let poseLoading = false;
let poseData = new Map();
let poseAnalysisRange = null;

// ============================================================================
// 1. 基礎角度與運算工具
// ============================================================================

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

// ============================================================================
// 2. 影片載入與渲染核心（已修復黑屏）
// ============================================================================

function loadVideo(file) {
  if (!file) return;
  if (videoUrl) URL.revokeObjectURL(videoUrl);
  
  videoUrl = URL.createObjectURL(file);
  poseData = new Map();
  takeoffFrame = null;
  landingFrame = null;
  startFrame = null;
  finishFrame = null;
  
  // 強制重設畫面狀態，確保影片元素顯示
  videoPreview.pause();
  videoPreview.removeAttribute("src");
  videoPreview.load();
  
  videoPreview.src = videoUrl;
  videoPreview.hidden = false;
  emptyVideo.hidden = true;
  videoOverlay.hidden = false;
  
  $("#videoName").textContent = file.name;
  analysisStatus.className = "analysis-status";
  analysisStatus.textContent = "影片已載入，正在讀取中繼資料...";
  
  videoPreview.addEventListener("loadedmetadata", () => {
    totalFrames = Math.max(1, Math.round(videoPreview.duration * Number(fpsSelect.value)));
    frameRange.max = String(totalFrames);
    clipRange.max = String(totalFrames);
    
    // 啟用控制元件
    [frameRange, clipRange, $("#previousFrame"), $("#nextFrame"), $("#playPause"), autoPoseButton].forEach((el) => {
      if (el) el.disabled = false;
    });
    
    $("#durationLabel").textContent = formatTime(videoPreview.duration || 0);
    updateReadouts();
    
    analysisStatus.className = "analysis-status success";
    analysisStatus.textContent = "影片已準備就緒，可開始播放或執行 AI 骨骼分析。";
  }, { once: true });

  videoPreview.addEventListener("error", (e) => {
    console.error("Video element error:", videoPreview.error);
    analysisStatus.className = "analysis-status error";
    analysisStatus.textContent = "影片載入失敗：瀏覽器無法解碼此格式，請嘗試更換標準 MP4 檔案。";
  }, { once: true });
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return "00:00.000";
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remaining.toFixed(3).padStart(6, "0")}`;
}

function updateReadouts() {
  const fps = Number(fpsSelect.value);
  const current = Number(frameRange.value || 0);
  readoutS.textContent = `S ${startFrame === null ? "—" : formatTime(startFrame / fps)}`;
  takeoffReadout.textContent = `T ${takeoffFrame === null ? "—" : formatTime(takeoffFrame / fps)}`;
  landingReadout.textContent = `L ${landingFrame === null ? "—" : formatTime(landingFrame / fps)}`;
  readoutF.textContent = `F ${finishFrame === null ? "—" : formatTime(finishFrame / fps)}`;
  $("#frameLabel").textContent = `Frame ${current} / ${totalFrames || 1}`;
  $("#videoTime").textContent = formatTime(videoPreview.currentTime || current / fps);
}

// ============================================================================
// 3. 事件綁定區
// ============================================================================

safetyConfirm.addEventListener("change", () => { enterApp.disabled = !safetyConfirm.checked; });
enterApp.addEventListener("click", () => { safetyModal.classList.add("hidden"); setAppEnabled(true); });
$("#openGuide").addEventListener("click", () => guideModal.classList.remove("hidden"));
$("#closeGuide").addEventListener("click", () => guideModal.classList.add("hidden"));

$("#uploadButton").addEventListener("click", () => videoInput.click());
$("#heroUpload").addEventListener("click", () => { document.querySelector("#analyzer").scrollIntoView({ behavior: "smooth" }); videoInput.click(); });
videoInput.addEventListener("change", (event) => loadVideo(event.target.files[0]));

frameRange.addEventListener("input", () => {
  const frame = Number(frameRange.value || 0);
  if (videoPreview.duration) videoPreview.currentTime = frame / Number(fpsSelect.value);
  updateReadouts();
});

videoPreview.addEventListener("timeupdate", () => {
  if (!videoPreview.seeking) {
    const currentFrame = Math.round(videoPreview.currentTime * Number(fpsSelect.value));
    frameRange.value = String(currentFrame);
    updateReadouts();
  }
});

$("#previousFrame").addEventListener("click", () => { frameRange.value = String(Math.max(0, Number(frameRange.value) - 1)); frameRange.dispatchEvent(new Event("input")); });
$("#nextFrame").addEventListener("click", () => { frameRange.value = String(Math.min(totalFrames, Number(frameRange.value) + 1)); frameRange.dispatchEvent(new Event("input")); });
$("#playPause").addEventListener("click", () => {
  if (videoPreview.paused) { videoPreview.play(); $("#playPause").innerHTML = "Ⅱ <span>暫停</span>"; }
  else { videoPreview.pause(); $("#playPause").innerHTML = "▶ <span>播放</span>"; }
});

function setAppEnabled(enabled) {
  [videoInput, $("#testType"), fpsSelect, $("#uploadButton"), $("#heroUpload")].forEach((element) => {
    if (element) element.disabled = !enabled;
  });
}

safetyModal.classList.add("hidden");
setAppEnabled(true);

