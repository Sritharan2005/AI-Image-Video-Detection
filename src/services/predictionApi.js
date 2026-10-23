/**
 * DeepGuard AI Prediction Service
 * 
 * Supports:
 * 1. DeepGuard FastAPI Backend (Default: EfficientNet-B0 Baseline & Face-Crop, Port 8000)
 * 2. DeepGuard Video Pipeline (Frame Sampling & Aggregation, Port 8000)
 * 3. AI Image Detector (Optional Experimental Ultra Ensemble, Port 8005)
 * 
 * Does NOT fabricate predictions, heuristic noise patterns, or hardcoded results.
 */

const API_BASE_URL = (typeof process !== 'undefined' && process.env?.VITE_API_URL) || (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL) || 'http://127.0.0.1:8000';
const DETECTOR_API_URL = (typeof process !== 'undefined' && process.env?.VITE_DETECTOR_API_URL) || (typeof import.meta !== 'undefined' && import.meta.env?.VITE_DETECTOR_API_URL) || 'http://127.0.0.1:8005';

/**
 * Checks connectivity and model status on the primary DeepGuard FastAPI backend (port 8000).
 * Uses an AbortController timeout to prevent hanging when services are starting up.
 *
 * @param {number} [timeoutMs=3500] - Request timeout in milliseconds.
 */
export async function checkBackendStatus(timeoutMs = 3500) {
  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timeoutId = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;

  try {
    const response = await fetch(`${API_BASE_URL}/api/status`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller?.signal,
    });
    if (timeoutId) clearTimeout(timeoutId);
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    if (timeoutId) clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error(`Connection timed out after ${timeoutMs}ms connecting to DeepGuard backend at ${API_BASE_URL}.`);
    }
    throw new Error(
      `Cannot connect to FastAPI backend at ${API_BASE_URL}. Ensure the server is running with 'python backend/main.py'.`
    );
  }
}

/**
 * Checks connectivity on the optional experimental AI Image Detector API (port 8005).
 * Uses an AbortController timeout to prevent hanging when services are starting up.
 *
 * @param {number} [timeoutMs=3500] - Request timeout in milliseconds.
 */
export async function checkDetectorStatus(timeoutMs = 3500) {
  const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
  const timeoutId = controller ? setTimeout(() => controller.abort(), timeoutMs) : null;

  try {
    const response = await fetch(`${DETECTOR_API_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller?.signal,
    });
    if (timeoutId) clearTimeout(timeoutId);
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    if (timeoutId) clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error(`Connection timed out after ${timeoutMs}ms connecting to Ultra detector at ${DETECTOR_API_URL}.`);
    }
    throw new Error(
      `Cannot connect to optional AI Image Detector at ${DETECTOR_API_URL}. If using Ultra mode, ensure the service is running on port 8005.`
    );
  }
}

/**
 * Asynchronously checks health of all backend services (ports 8000 and 8005).
 * Completely non-blocking with strict timeouts.
 *
 * @param {Object} [options]
 * @param {number} [options.timeoutMs=3500] - Request timeout per health check in milliseconds.
 * @returns {Promise<Object>} Aggregated readiness status.
 */
export async function checkServicesHealth(options = {}) {
  const timeoutMs = options.timeoutMs || 3500;
  const [deepguardRes, ultraRes] = await Promise.allSettled([
    checkBackendStatus(timeoutMs),
    checkDetectorStatus(timeoutMs),
  ]);

  const deepguardOk = deepguardRes.status === 'fulfilled' && !!deepguardRes.value;
  const ultraOk = ultraRes.status === 'fulfilled' && (ultraRes.value?.status === 'ok' || !ultraRes.value?.error);

  const readyCount = (deepguardOk ? 2 : 0) + (ultraOk ? 1 : 0);

  return {
    allReady: deepguardOk && ultraOk,
    deepguard: {
      ready: deepguardOk,
      error: deepguardRes.status === 'rejected' ? deepguardRes.reason?.message : null,
      data: deepguardOk ? deepguardRes.value : null,
    },
    ultra: {
      ready: ultraOk,
      error: ultraRes.status === 'rejected' ? ultraRes.reason?.message : null,
      data: ultraOk ? ultraRes.value : null,
    },
    detectorsReadyCount: readyCount,
    totalDetectors: 3,
  };
}

/**
 * Converts an image URL (such as a local sample card image) to a standard File object.
 */
async function urlToFile(url, filename = 'sample.jpg') {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load sample image from ${url}`);
  }
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || 'image/jpeg' });
}

/**
 * Calls the optional experimental ai-image-detector Ultra backend API (POST /detect on port 8005).
 * 
 * @param {File|Blob|string} fileOrUrl - The uploaded image file or URL.
 * @returns {Promise<Object>} The detection result from the Ultra detector.
 */
export async function detectWithUltra(fileOrUrl) {
  let filePayload = fileOrUrl;
  if (typeof fileOrUrl === 'string') {
    const filename = fileOrUrl.split('/').pop() || 'sample.jpg';
    filePayload = await urlToFile(fileOrUrl, filename);
  }

  if (!filePayload || !(filePayload instanceof Blob || filePayload instanceof File)) {
    throw new Error('Invalid file payload. Please select a valid image file.');
  }

  const formData = new FormData();
  formData.append('file', filePayload, filePayload.name || 'uploaded_image.jpg');

  let response;
  try {
    response = await fetch(`${DETECTOR_API_URL}/detect`, {
      method: 'POST',
      body: formData,
    });
  } catch (networkErr) {
    console.error('[AI Detector API] Connection failed:', networkErr);
    throw new Error(
      `Optional AI Detector Unavailable: Unable to connect to the Ultra Detector service at ${DETECTOR_API_URL}. ` +
      'Please verify that the service is running: `.\\.venv\\Scripts\\aidetect.exe api --host 127.0.0.1 --port 8005 --backend ultra` ' +
      'or switch to DeepGuard Baseline (port 8000).'
    );
  }

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch (_) {}
    throw new Error(`AI Detector Error: ${errorDetail}`);
  }

  const data = await response.json();
  const isReal = data.label?.toLowerCase() === 'real';

  return {
    media_type: 'image',
    prediction: isReal ? 'REAL' : 'AI GENERATED',
    label: data.label,
    probability_ai: data.probability_ai,
    probability_real: data.probability_real,
    confidence: data.confidence,
    raw_score: data.raw_score,
    backend: data.backend,
    model_version: `AI Image Detector (${data.backend?.toUpperCase() || 'ULTRA'})`,
    source: `AI Image Detector API (${DETECTOR_API_URL}/detect)`,
    probabilities: {
      REAL: data.probability_real,
      FAKE: data.probability_ai,
    },
    raw_probabilities: {
      REAL: data.probability_real,
      FAKE: data.probability_ai,
    },
    explanation: isReal
      ? `This media was classified as REAL by the experimental ${data.backend?.toUpperCase() || 'ULTRA'} ensemble based on natural semantic features and frequency residuals.`
      : `This media exhibits generative characteristics consistent with AI synthesis identified by the experimental ${data.backend?.toUpperCase() || 'ULTRA'} ensemble.`,
    decision: {
      operating_mode: `Experimental Ultra Ensemble (${data.backend?.toUpperCase() || 'ULTRA'})`,
      threshold_applied: 0.50,
      backend: data.backend,
      raw_score: data.raw_score,
    },
    disclaimer: 'AI image detection is probabilistic and should not be used as sole proof. Highly realistic AI generations may not display detectable artifacts.',
  };
}

/**
 * Main inference API call for image or video files.
 * Defaults to DeepGuard Baseline (port 8000) for images.
 * 
 * @param {File|Blob|string} fileOrUrl - The uploaded image file or URL.
 * @param {boolean} isVideo - Flag indicating whether the file is a video.
 * @param {string} mode - 'baseline' (Default, threshold 0.40) | 'experimental_face' (threshold 0.35) | 'ultra_detector' (Optional Port 8005)
 * @param {number|null} customThreshold - Optional custom decision threshold in [0.0, 1.0].
 * @returns {Promise<Object>} The genuine inference results.
 */
/**
 * Experimental Ensemble Voting Mode:
 * Queries DeepGuard Baseline (Port 8000), DeepGuard Face-Crop (Port 8000),
 * and Ultra Detector (Port 8005) simultaneously.
 * Applies transparent 2-of-3 majority voting. If a detector fails, prediction is missing,
 * or no majority is reached, returns UNCERTAIN.
 *
 * @param {File|Blob} filePayload - Image file to evaluate.
 * @returns {Promise<Object>} Unified ensemble voting result with individual breakdowns.
 */
export async function detectWithEnsembleVoting(filePayload) {
  if (!filePayload || !(filePayload instanceof Blob || filePayload instanceof File)) {
    throw new Error('Invalid file payload for ensemble voting.');
  }

  // 1. Helper for DeepGuard Baseline (Port 8000)
  const fetchBaseline = async () => {
    const formData = new FormData();
    formData.append('file', filePayload, filePayload.name || 'uploaded_image.jpg');
    const res = await fetch(`${API_BASE_URL}/predict?mode=baseline`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      let errDetail = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errDetail;
      } catch (_) {}
      throw new Error(errDetail);
    }
    return await res.json();
  };

  // 2. Helper for DeepGuard Face-Crop (Port 8000)
  const fetchFaceCrop = async () => {
    const formData = new FormData();
    formData.append('file', filePayload, filePayload.name || 'uploaded_image.jpg');
    const res = await fetch(`${API_BASE_URL}/predict?mode=experimental_face`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      let errDetail = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errDetail;
      } catch (_) {}
      throw new Error(errDetail);
    }
    return await res.json();
  };

  // 3. Helper for Ultra Detector (Port 8005)
  const fetchUltra = async () => {
    const formData = new FormData();
    formData.append('file', filePayload, filePayload.name || 'uploaded_image.jpg');
    const res = await fetch(`${DETECTOR_API_URL}/detect`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      let errDetail = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        errDetail = errJson.detail || errDetail;
      } catch (_) {}
      throw new Error(errDetail);
    }
    return await res.json();
  };

  // Dispatch all 3 requests in parallel
  const [baselineRes, faceCropRes, ultraRes] = await Promise.allSettled([
    fetchBaseline(),
    fetchFaceCrop(),
    fetchUltra(),
  ]);

  // Baseline normalization
  const baselineData = baselineRes.status === 'fulfilled' ? baselineRes.value : null;
  const baselineErr = baselineRes.status === 'rejected' ? (baselineRes.reason?.message || 'Request failed') : null;
  const baselineLabel = baselineData ? (baselineData.prediction === 'FAKE' ? 'FAKE' : (baselineData.prediction === 'REAL' ? 'REAL' : null)) : null;

  // Face-Crop normalization
  const faceCropData = faceCropRes.status === 'fulfilled' ? faceCropRes.value : null;
  const faceCropErr = faceCropRes.status === 'rejected' ? (faceCropRes.reason?.message || 'Request failed') : null;
  const faceCropLabel = faceCropData ? (faceCropData.prediction === 'FAKE' ? 'FAKE' : (faceCropData.prediction === 'REAL' ? 'REAL' : null)) : null;

  // Ultra normalization
  const ultraData = ultraRes.status === 'fulfilled' ? ultraRes.value : null;
  const ultraErr = ultraRes.status === 'rejected' ? (ultraRes.reason?.message || 'Request failed') : null;
  let ultraLabel = null;
  if (ultraData && ultraData.label) {
    const l = String(ultraData.label).toLowerCase();
    if (l === 'ai' || l === 'fake') ultraLabel = 'FAKE';
    else if (l === 'real') ultraLabel = 'REAL';
  }

  const detectors = [
    {
      id: 'baseline',
      name: 'DeepGuard Baseline',
      port: 8000,
      architecture: baselineData?.model_version || 'EfficientNet-B0 (Full Image)',
      status: baselineErr ? 'error' : 'success',
      error: baselineErr,
      label: baselineLabel,
      probability_ai: baselineData?.probabilities?.FAKE ?? baselineData?.raw_probabilities?.FAKE ?? null,
      probability_real: baselineData?.probabilities?.REAL ?? baselineData?.raw_probabilities?.REAL ?? null,
      confidence: baselineData?.confidence ?? null,
    },
    {
      id: 'face_crop',
      name: 'DeepGuard Face-Crop',
      port: 8000,
      architecture: faceCropData?.model_version || 'EfficientNet-B0 (Face Crop)',
      status: faceCropErr ? 'error' : 'success',
      error: faceCropErr,
      label: faceCropLabel,
      face_detected: faceCropData?.face_detected ?? false,
      probability_ai: faceCropData?.probabilities?.FAKE ?? faceCropData?.raw_probabilities?.FAKE ?? null,
      probability_real: faceCropData?.probabilities?.REAL ?? faceCropData?.raw_probabilities?.REAL ?? null,
      confidence: faceCropData?.confidence ?? null,
    },
    {
      id: 'ultra',
      name: 'Ultra Detector',
      port: 8005,
      architecture: 'UnivFD + Sentry + Nonescape + HF Ensemble',
      status: ultraErr ? 'error' : 'success',
      error: ultraErr,
      label: ultraLabel,
      probability_ai: ultraData?.probability_ai ?? null,
      probability_real: ultraData?.probability_real ?? null,
      confidence: ultraData?.confidence ?? null,
      raw_score: ultraData?.raw_score ?? null,
    },
  ];

  return aggregateEnsembleVoting(detectors);
}

/**
 * Deterministically computes the Quadratic Mean (Root Mean Square - RMS) over winning detector confidences.
 * Formula: S_RMS = sqrt((1/k) * sum(c_i^2))
 *
 * @param {number[]} confidences - Array of numeric confidence values in [0.0, 1.0].
 * @returns {number} The RMS aggregated confidence score.
 */
export function computeRmsConfidence(confidences) {
  if (!Array.isArray(confidences) || confidences.length === 0) return 0.0;
  const k = confidences.length;
  const sumSquares = confidences.reduce((sum, c) => sum + (c * c), 0);
  return Math.sqrt(sumSquares / k);
}

/**
 * Pure deterministic voting aggregation function for 3-way DeepGuard ensemble.
 * 
 * Rules:
 * 1. Requires exactly 3 detector evaluations.
 * 2. If any detector failed (status !== 'success'), label is missing, NaN, or out of range [0, 1], returns UNCERTAIN (confidence 0.0).
 * 3. Majority rule: >= 2 FAKE -> FAKE; >= 2 REAL -> REAL; otherwise UNCERTAIN.
 * 4. Dynamic confidence is computed via Root Mean Square (RMS) strictly over the winning detectors' confidence in the winning class:
 *    - FAKE majority: P(FAKE) / probability_ai
 *    - REAL majority: P(REAL) / probability_real
 *
 * @param {Array<Object>} detectors - Array of 3 normalized detector output objects.
 * @returns {Object} Unified ensemble decision payload.
 */
export function aggregateEnsembleVoting(detectors) {
  if (!Array.isArray(detectors) || detectors.length !== 3) {
    return {
      media_type: 'image',
      is_ensemble: true,
      prediction: 'UNCERTAIN',
      label: 'UNCERTAIN',
      confidence: 0.0,
      vote_counts: { FAKE: 0, REAL: 0, UNCERTAIN: 3 },
      majority_reached: false,
      individual_detectors: detectors || [],
      probabilities: { REAL: 0.5, FAKE: 0.5 },
      raw_probabilities: { REAL: 0.5, FAKE: 0.5 },
      model_version: '3-Way Ensemble (DeepGuard Baseline + Face-Crop + Ultra)',
      source: 'Ensemble Voting Engine (Ports 8000 & 8005)',
      explanation: 'Ensemble result is UNCERTAIN: Expected exactly 3 detector evaluations.',
      decision: {
        operating_mode: 'Experimental Ensemble Voting (Majority 2/3)',
        threshold_applied: 'Decision Rule: 2 of 3 votes',
        decision_rule: '2 of 3 votes',
        aggregation_method: 'Root Mean Square (RMS)',
        aggregation_formula: 'sqrt((1/k) * sum(c_i^2)) over winning detectors',
        majority_reached: false,
        votes: { FAKE: 0, REAL: 0, UNCERTAIN: 3 },
        winning_class: 'UNCERTAIN',
        winning_detectors_count: 0,
        winning_detector_confidences: [],
        ensemble_confidence_score: 0.0,
      },
      disclaimer: 'Ensemble voting is an experimental aggregation rule. Majority voting does not guarantee accuracy improvement without empirical benchmark evaluation on held-out datasets.',
    };
  }

  // Voting Rule:
  // 1. Voting determines the label first:
  //    At least 2 FAKE votes -> final label FAKE.
  //    At least 2 REAL votes -> final label REAL.
  //    If any detector fails, output is invalid, or no majority -> UNCERTAIN.
  const hasFailure = detectors.some(d => d.status !== 'success' || !d.label || (d.label !== 'FAKE' && d.label !== 'REAL'));

  // Validate probability bounds [0.0, 1.0]
  const hasInvalidProb = detectors.some(d => {
    if (d.status !== 'success') return false;
    const pAiValid = typeof d.probability_ai === 'number' && !isNaN(d.probability_ai) && d.probability_ai >= 0.0 && d.probability_ai <= 1.0;
    const pRealValid = typeof d.probability_real === 'number' && !isNaN(d.probability_real) && d.probability_real >= 0.0 && d.probability_real <= 1.0;
    return !pAiValid || !pRealValid;
  });

  let finalPrediction = 'UNCERTAIN';
  const voteCounts = { FAKE: 0, REAL: 0, UNCERTAIN: 0 };
  let explanation = '';
  let majorityReached = false;
  let ensembleConfidence = 0.0;
  let winningConfidences = [];

  if (hasFailure) {
    const failedNames = detectors.filter(d => d.status !== 'success' || !d.label || (d.label !== 'FAKE' && d.label !== 'REAL')).map(d => `${d.name || d.id || 'Detector'} (${d.port || 'unknown'})`).join(', ');
    finalPrediction = 'UNCERTAIN';
    explanation = `Ensemble result is UNCERTAIN because one or more detector services failed or returned an incomplete response (${failedNames}). Voting requires all 3 detectors to participate without invented values.`;
  } else if (hasInvalidProb) {
    finalPrediction = 'UNCERTAIN';
    explanation = 'Ensemble result is UNCERTAIN because one or more detectors returned invalid or out-of-bounds probability values.';
  } else {
    detectors.forEach(d => {
      if (d.label === 'FAKE') voteCounts.FAKE += 1;
      else if (d.label === 'REAL') voteCounts.REAL += 1;
      else voteCounts.UNCERTAIN += 1;
    });

    if (voteCounts.FAKE >= 2) {
      finalPrediction = 'FAKE';
      majorityReached = true;
    } else if (voteCounts.REAL >= 2) {
      finalPrediction = 'REAL';
      majorityReached = true;
    } else {
      finalPrediction = 'UNCERTAIN';
      explanation = `Ensemble decision is UNCERTAIN: No label achieved a 2-of-3 majority (FAKE: ${voteCounts.FAKE}, REAL: ${voteCounts.REAL}).`;
    }

    if (majorityReached) {
      // 2. Dynamic confidence score:
      // Extract the winning detectors' actual class probabilities
      const winningDetectors = detectors.filter(d => d.label === finalPrediction);
      winningConfidences = winningDetectors.map(d => {
        return finalPrediction === 'FAKE' ? d.probability_ai : d.probability_real;
      });

      // Deterministic Aggregation Formula: Quadratic Mean (Root Mean Square - RMS)
      // S_RMS = sqrt((1/k) * sum(c_i^2)) over the k winning detectors (k in {2, 3})
      // - Varies dynamically when underlying confidences change, even at 2/3 votes
      // - Is not a fixed 66.7% vote share
      // - Is not a simple arithmetic mean
      // - Does not artificially inflate or force FAKE higher than REAL
      const k = winningConfidences.length;
      ensembleConfidence = computeRmsConfidence(winningConfidences);

      explanation = `Ensemble decision is ${finalPrediction} (${k} of 3 votes). Dynamic Ensemble Confidence Score: ${(ensembleConfidence * 100).toFixed(1)}% (computed via Root Mean Square over winning models with individual scores: ${winningConfidences.map(c => (c * 100).toFixed(1) + '%').join(', ')}).`;
    }
  }

  // Mean probabilities of responding detectors for reference
  const validAi = detectors.filter(d => typeof d.probability_ai === 'number' && !isNaN(d.probability_ai)).map(d => d.probability_ai);
  const validReal = detectors.filter(d => typeof d.probability_real === 'number' && !isNaN(d.probability_real)).map(d => d.probability_real);
  const meanAi = validAi.length ? (validAi.reduce((a, b) => a + b, 0) / validAi.length) : (finalPrediction === 'FAKE' ? 1.0 : (finalPrediction === 'REAL' ? 0.0 : 0.5));
  const meanReal = validReal.length ? (validReal.reduce((a, b) => a + b, 0) / validReal.length) : (1.0 - meanAi);

  return {
    media_type: 'image',
    is_ensemble: true,
    prediction: finalPrediction,
    label: finalPrediction,
    confidence: ensembleConfidence,
    vote_counts: voteCounts,
    majority_reached: majorityReached,
    individual_detectors: detectors,
    probabilities: {
      REAL: meanReal,
      FAKE: meanAi,
    },
    raw_probabilities: {
      REAL: meanReal,
      FAKE: meanAi,
    },
    model_version: '3-Way Ensemble (DeepGuard Baseline + Face-Crop + Ultra)',
    source: 'Ensemble Voting Engine (Ports 8000 & 8005)',
    explanation: explanation,
    decision: {
      operating_mode: 'Experimental Ensemble Voting (Majority 2/3)',
      threshold_applied: 'Decision Rule: 2 of 3 votes',
      decision_rule: '2 of 3 votes',
      aggregation_method: 'Root Mean Square (RMS)',
      aggregation_formula: 'sqrt((1/k) * sum(c_i^2)) over winning detectors',
      majority_reached: majorityReached,
      votes: voteCounts,
      winning_class: finalPrediction,
      winning_detectors_count: majorityReached ? winningConfidences.length : 0,
      winning_detector_confidences: winningConfidences,
      ensemble_confidence_score: ensembleConfidence,
    },
    disclaimer: 'Ensemble voting is an experimental aggregation rule. Majority voting does not guarantee accuracy improvement without empirical benchmark evaluation on held-out datasets.',
  };
}


export async function predictFile(
  fileOrUrl,
  isVideo = false,
  mode = 'baseline',
  customThreshold = null,
  options = {}
) {
  // If input is a URL string (from sample card), convert to File object
  let filePayload = fileOrUrl;
  if (typeof fileOrUrl === 'string') {
    const filename = fileOrUrl.split('/').pop() || (isVideo ? 'sample.mp4' : 'sample.jpg');
    filePayload = await urlToFile(fileOrUrl, filename);
  }

  if (!filePayload || !(filePayload instanceof Blob || filePayload instanceof File)) {
    throw new Error(
      isVideo
        ? 'Invalid file payload. Please select a valid video file.'
        : 'Invalid file payload. Please select a valid image file.'
    );
  }

  // 1. Optional Experimental Ultra Detector Mode (Port 8005)
  if (!isVideo && (mode === 'ultra_detector' || mode === 'ultra')) {
    return await detectWithUltra(filePayload);
  }

  // 2. Optional Experimental Ensemble Voting Mode (Ports 8000 & 8005)
  if (!isVideo && (mode === 'ensemble_voting' || mode === 'ensemble')) {
    return await detectWithEnsembleVoting(filePayload);
  }

  // 2. Primary DeepGuard Backend (Port 8000) - Images (Baseline, Face-Crop) & Videos
  const formData = new FormData();
  formData.append('file', filePayload, filePayload.name || (isVideo ? 'uploaded_video.mp4' : 'uploaded_image.jpg'));

  const params = new URLSearchParams();

  // Video Pipeline
  if (isVideo) {
    if (customThreshold !== null && customThreshold !== undefined) {
      params.append('threshold', String(customThreshold));
    }
    const maxFrames = options.maxFrames || 8;
    params.append('max_frames', String(maxFrames));
    const aggregation = options.aggregation || 'mean';
    params.append('aggregation', aggregation);

    const endpointUrl = `${API_BASE_URL}/api/predict/video?${params.toString()}`;

    let response;
    try {
      response = await fetch(endpointUrl, {
        method: 'POST',
        body: formData,
      });
    } catch (networkErr) {
      console.error('[DeepGuard API] Connection failed:', networkErr);
      throw new Error(
        `Backend Unavailable: Unable to connect to DeepGuard FastAPI service at ${API_BASE_URL}. ` +
        'Please verify that the backend is running locally on port 8000 (`python backend/main.py`).'
      );
    }

    if (!response.ok) {
      let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errJson = await response.json();
        errorDetail = errJson.detail || errJson.message || errorDetail;
      } catch (_) {}

      if (response.status === 400) {
        throw new Error(`Invalid Video: ${errorDetail}`);
      } else if (response.status === 413) {
        throw new Error(`Video Too Large: ${errorDetail}`);
      } else if (response.status === 422) {
        throw new Error(`Unprocessable Video: ${errorDetail}`);
      } else if (response.status === 503) {
        throw new Error(`Model Unavailable: ${errorDetail}`);
      } else {
        throw new Error(`Video Inference Service Error: ${errorDetail}`);
      }
    }

    const data = await response.json();
    return {
      ...data,
      source: `FastAPI Backend (${API_BASE_URL}/api/predict/video)`,
      disclaimer: 'Frame-level static evaluations. Does not measure temporal consistency or video compression artifacts.',
    };
  }

  // Image Pipeline (Default: DeepGuard Baseline or Face-Crop on Port 8000)
  if (mode === 'experimental_face') {
    params.append('mode', 'experimental_face');
  } else if (mode === 'experimental_multi_crop' || mode === 'multi_crop') {
    params.append('mode', 'experimental_multi_crop');
  } else {
    params.append('mode', 'baseline');
  }

  if (customThreshold !== null && customThreshold !== undefined) {
    params.append('threshold', String(customThreshold));
  }

  const endpointUrl = `${API_BASE_URL}/predict?${params.toString()}`;

  let response;
  try {
    response = await fetch(endpointUrl, {
      method: 'POST',
      body: formData,
    });
  } catch (networkErr) {
    console.error('[DeepGuard API] Connection failed:', networkErr);
    throw new Error(
      `Backend Unavailable: Unable to connect to DeepGuard FastAPI service at ${API_BASE_URL}. ` +
      'Please verify that the backend is running locally on port 8000 (`python backend/main.py`).'
    );
  }

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch (_) {}

    if (response.status === 400) {
      throw new Error(`Invalid File: ${errorDetail}`);
    } else if (response.status === 413) {
      throw new Error(`File Too Large: ${errorDetail}`);
    } else if (response.status === 422) {
      throw new Error(`Unprocessable Image: ${errorDetail}`);
    } else if (response.status === 503) {
      throw new Error(`Model Unavailable: ${errorDetail}`);
    } else {
      throw new Error(`Inference Service Error: ${errorDetail}`);
    }
  }

  const data = await response.json();

  return {
    ...data,
    source: `DeepGuard FastAPI Backend (${API_BASE_URL}/predict)`,
    disclaimer: 'AI image detection is probabilistic and should not be used as sole proof. Highly realistic AI generations may not display detectable artifacts.',
  };
}