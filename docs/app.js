const CARD_WIDTH_MM = 85.6;
const CARD_HEIGHT_MM = 54;
const CARD_WIDTH_PX = 1011;
const CARD_HEIGHT_PX = 638;
const A4_WIDTH_PT = 595.2756;
const A4_HEIGHT_PT = 841.8898;
const MM_TO_PT = 72 / 25.4;

const statusBox = document.getElementById("status");
const submitButton = document.getElementById("submit");
const files = { front: null, back: null };

function setStatus(message, kind = "") {
  statusBox.textContent = message;
  statusBox.className = `alert ${kind}`.trim();
}

function orderPoints(points) {
  const sum = (point) => point[0] + point[1];
  const diff = (point) => point[1] - point[0];
  return [
    points.reduce((a, b) => (sum(a) < sum(b) ? a : b)),
    points.reduce((a, b) => (diff(a) < diff(b) ? a : b)),
    points.reduce((a, b) => (sum(a) > sum(b) ? a : b)),
    points.reduce((a, b) => (diff(a) > diff(b) ? a : b)),
  ];
}

function imageToCanvas(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const scale = Math.min(1, 1600 / Math.max(image.naturalWidth, image.naturalHeight));
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(image.src);
      resolve(canvas);
    };
    image.onerror = () => reject(new Error("无法读取图片，请使用 JPG、PNG 或浏览器支持的图片格式。"));
    image.src = URL.createObjectURL(file);
  });
}

function detectCorners(source) {
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const edges = new cv.Mat();
  const dilated = new cv.Mat();
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.cvtColor(source, gray, cv.COLOR_RGBA2GRAY);
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
  cv.Canny(blurred, edges, 45, 140);
  cv.dilate(edges, dilated, cv.Mat.ones(5, 5, cv.CV_8U));
  cv.findContours(dilated, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const imageArea = source.rows * source.cols;
  let best = null;
  let largest = null;
  let largestArea = 0;
  for (let i = 0; i < contours.size(); i += 1) {
    const contour = contours.get(i);
    const area = Math.abs(cv.contourArea(contour));
    if (area > largestArea) {
      largestArea = area;
      largest = contour;
    }
    const perimeter = cv.arcLength(contour, true);
    const approx = new cv.Mat();
    cv.approxPolyDP(contour, approx, 0.02 * perimeter, true);
    if (approx.rows === 4 && area > imageArea * 0.08 && cv.isContourConvex(approx)) {
      const data = approx.data32S || approx.data32F;
      const points = [];
      for (let j = 0; j < 8; j += 2) points.push([data[j], data[j + 1]]);
      const score = area / imageArea;
      if (!best || score > best.score) best = { score, points };
    }
    approx.delete();
    contour.delete();
  }

  let points = best?.points;
  if (!points && largest && largestArea > imageArea * 0.05) {
    const rect = cv.boundingRect(largest);
    points = [[rect.x, rect.y], [rect.x + rect.width, rect.y], [rect.x + rect.width, rect.y + rect.height], [rect.x, rect.y + rect.height]];
  }
  gray.delete(); blurred.delete(); edges.delete(); dilated.delete(); contours.delete(); hierarchy.delete();
  if (!points) throw new Error("没有找到足够大的证件轮廓，请让证件完整出现在画面内。");
  return orderPoints(points);
}

function cropCard(canvas) {
  const source = cv.imread(canvas);
  const corners = detectCorners(source);
  const sourcePoints = cv.matFromArray(4, 1, cv.CV_32FC2, corners.flat());
  const targetPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, CARD_WIDTH_PX - 1, 0, CARD_WIDTH_PX - 1, CARD_HEIGHT_PX - 1, 0, CARD_HEIGHT_PX - 1]);
  const transform = cv.getPerspectiveTransform(sourcePoints, targetPoints);
  const output = new cv.Mat();
  cv.warpPerspective(source, output, transform, new cv.Size(CARD_WIDTH_PX, CARD_HEIGHT_PX), cv.INTER_LINEAR, cv.BORDER_REPLICATE, new cv.Scalar());
  const resultCanvas = document.createElement("canvas");
  resultCanvas.width = CARD_WIDTH_PX;
  resultCanvas.height = CARD_HEIGHT_PX;
  cv.imshow(resultCanvas, output);
  const dataUrl = resultCanvas.toDataURL("image/jpeg", 0.96);
  source.delete(); output.delete(); sourcePoints.delete(); targetPoints.delete(); transform.delete();
  return dataUrl;
}

async function makePdf(frontDataUrl, backDataUrl) {
  if (!window.PDFLib) throw new Error("PDF 引擎加载失败，请刷新页面重试。");
  const pdf = await PDFLib.PDFDocument.create();
  const page = pdf.addPage([A4_WIDTH_PT, A4_HEIGHT_PT]);
  const cardWidth = CARD_WIDTH_MM * MM_TO_PT;
  const cardHeight = CARD_HEIGHT_MM * MM_TO_PT;
  const margin = 20 * MM_TO_PT;
  const gap = 18 * MM_TO_PT;
  const x = (A4_WIDTH_PT - cardWidth) / 2;
  const topY = A4_HEIGHT_PT - margin - cardHeight;
  const bottomY = topY - gap - cardHeight;
  const front = await pdf.embedJpg(frontDataUrl);
  const back = await pdf.embedJpg(backDataUrl);
  page.drawImage(front, { x, y: topY, width: cardWidth, height: cardHeight });
  page.drawImage(back, { x, y: bottomY, width: cardWidth, height: cardHeight });
  return pdf.save();
}

function bindPreview(inputId, previewId, actionId, key) {
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const action = document.getElementById(actionId);
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) return;
    files[key] = file;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    action.textContent = "重新选择照片";
    submitButton.disabled = !(files.front && files.back);
  });
}

bindPreview("front", "frontPreview", "frontAction", "front");
bindPreview("back", "backPreview", "backAction", "back");

window.opencvReady.then(() => {
  setStatus("处理引擎已就绪，照片不会离开当前浏览器。", "ready");
  submitButton.disabled = !(files.front && files.back);
}).catch(() => setStatus("处理引擎加载失败，请刷新页面重试。", "error"));

submitButton.addEventListener("click", async () => {
  if (!files.front || !files.back) return;
  submitButton.disabled = true;
  setStatus("正在本地识别四角、矫正透视并生成 PDF……");
  try {
    await window.opencvReady;
    const front = cropCard(await imageToCanvas(files.front));
    const back = cropCard(await imageToCanvas(files.back));
    const bytes = await makePdf(front, back);
    const blob = new Blob([bytes], { type: "application/pdf" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14);
    link.download = `身份证-正反面-${stamp}.pdf`;
    link.click();
    URL.revokeObjectURL(link.href);
    setStatus("处理完成，PDF 已下载到你的设备。", "ready");
  } catch (error) {
    setStatus(error?.message || "处理失败，请换一张边缘清晰的照片重试。", "error");
  } finally {
    submitButton.disabled = !(files.front && files.back);
  }
});
