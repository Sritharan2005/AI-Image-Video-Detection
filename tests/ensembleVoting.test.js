import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeRmsConfidence,
  aggregateEnsembleVoting,
  checkBackendStatus,
  checkDetectorStatus,
  checkServicesHealth,
} from '../src/services/predictionApi.js';
import {
  calculateAdjustedDisplayScore,
  formatAdjustedDisplayScore,
} from '../src/components/displayScore.js';

function approxEqual(actual, expected, tolerance = 1e-4, msg = '') {
  const diff = Math.abs(actual - expected);
  assert.ok(
    diff <= tolerance,
    msg + ': Expected ' + expected + ' +/- ' + tolerance + ', but received ' + actual + ' (diff: ' + diff + ')'
  );
}

// ---------------------------------------------------------------------------
// 1. RMS Formula Deterministic Verification
// ---------------------------------------------------------------------------
test('RMS Formula: Unanimous FAKE (0.85, 0.90, 0.94)', () => {
  const values = [0.85, 0.90, 0.94];
  const rms = computeRmsConfidence(values);
  // Expected: sqrt((0.85^2 + 0.90^2 + 0.94^2) / 3) = sqrt(2.4161 / 3) = 0.8974222...
  approxEqual(rms, 0.8974222, 1e-5, 'Unanimous FAKE RMS');
  approxEqual(rms * 100, 89.7422, 1e-3, 'Percentage representation ~89.74%');
});

test('RMS Formula: Unanimous REAL (0.92, 0.94, 0.96)', () => {
  const values = [0.92, 0.94, 0.96];
  const rms = computeRmsConfidence(values);
  // Expected: sqrt((0.92^2 + 0.94^2 + 0.96^2) / 3) = sqrt(2.6516 / 3) = 0.9401418...
  approxEqual(rms, 0.9401418, 1e-5, 'Unanimous REAL RMS');
  approxEqual(rms * 100, 94.0142, 1e-3, 'Percentage representation ~94.01%');
});

test('RMS Formula: Suite A FAKE Pairs (k=2)', () => {
  // Low: (0.54, 0.58) -> sqrt((0.2916 + 0.3364)/2) = sqrt(0.314) = 0.560357
  approxEqual(computeRmsConfidence([0.54, 0.58]), 0.560357, 1e-5);
  // Moderate: (0.70, 0.75) -> sqrt((0.49 + 0.5625)/2) = sqrt(0.52625) = 0.725431
  approxEqual(computeRmsConfidence([0.70, 0.75]), 0.725431, 1e-5);
  // High: (0.92, 0.96) -> sqrt((0.8464 + 0.9216)/2) = sqrt(0.884) = 0.940213
  approxEqual(computeRmsConfidence([0.92, 0.96]), 0.940213, 1e-5);
});

test('RMS Formula: Suite B REAL Pairs (k=2)', () => {
  // Borderline: (0.56, 0.62) -> sqrt((0.3136 + 0.3844)/2) = sqrt(0.349) = 0.590762
  approxEqual(computeRmsConfidence([0.56, 0.62]), 0.590762, 1e-5);
  // Solid: (0.82, 0.88) -> sqrt((0.6724 + 0.7744)/2) = sqrt(0.7234) = 0.850529
  approxEqual(computeRmsConfidence([0.82, 0.88]), 0.850529, 1e-5);
  // High: (0.95, 0.98) -> sqrt((0.9025 + 0.9604)/2) = sqrt(0.93145) = 0.965117
  approxEqual(computeRmsConfidence([0.95, 0.98]), 0.965117, 1e-5);
});

test('RMS Formula: Live Image Sample Cases', () => {
  // sample-fake-woman.jpg: [0.5237, 0.6004] -> 0.563357 (56.34% rounded, 56.3% to 1 decimal)
  approxEqual(computeRmsConfidence([0.5237, 0.6004]), 0.563357, 1e-5);
  // ai-generated.png: [0.7685, 0.6476] -> 0.710626
  approxEqual(computeRmsConfidence([0.7685, 0.6476]), 0.710626, 1e-5);
  // human.jpeg: [0.9239, 0.8313] -> 0.878820
  approxEqual(computeRmsConfidence([0.9239, 0.8313]), 0.878820, 1e-5);
  // ai_retouched.png: [0.7827, 0.8514] -> 0.817772
  approxEqual(computeRmsConfidence([0.7827, 0.8514]), 0.817772, 1e-5);
  // hero-face-fake.jpg: [0.9775, 0.7901, 0.5213] -> 0.785603
  approxEqual(computeRmsConfidence([0.9775, 0.7901, 0.5213]), 0.785603, 1e-5);
  // sample-fake-politician.jpg: [0.8797, 0.6687, 0.9435] -> 0.838893
  approxEqual(computeRmsConfidence([0.8797, 0.6687, 0.9435]), 0.838893, 1e-5);
  // sample-real-man.jpg: [0.6621, 0.6658, 0.8681] -> 0.738301
  approxEqual(computeRmsConfidence([0.6621, 0.6658, 0.8681]), 0.738301, 1e-5);
  // sample-real-landscape.jpg: [0.9696, 0.9696, 0.8531] -> 0.932385
  approxEqual(computeRmsConfidence([0.9696, 0.9696, 0.8531]), 0.932385, 1e-5);
});

// ---------------------------------------------------------------------------
// 2. Majority Rule & Winning Class Probability Verification
// ---------------------------------------------------------------------------
test('Majority FAKE (2 of 3 votes): takes P(FAKE) from winning detectors', () => {
  const detectors = [
    {
      id: 'baseline',
      name: 'Baseline',
      status: 'success',
      label: 'FAKE',
      probability_ai: 0.85,
      probability_real: 0.15,
    },
    {
      id: 'face_crop',
      name: 'Face-Crop',
      status: 'success',
      label: 'FAKE',
      probability_ai: 0.91,
      probability_real: 0.09,
    },
    {
      id: 'ultra',
      name: 'Ultra',
      status: 'success',
      label: 'REAL',
      probability_ai: 0.20,
      probability_real: 0.80,
    },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'FAKE');
  assert.equal(result.label, 'FAKE');
  assert.equal(result.majority_reached, true);
  assert.deepEqual(result.vote_counts, { FAKE: 2, REAL: 1, UNCERTAIN: 0 });
  assert.deepEqual(result.decision.winning_detector_confidences, [0.85, 0.91]);

  // Expected RMS of [0.85, 0.91] = sqrt((0.7225 + 0.8281) / 2) = sqrt(0.7753) = 0.880511...
  approxEqual(result.confidence, 0.880511, 1e-5);
});

test('Majority REAL (2 of 3 votes): takes P(REAL) from winning detectors', () => {
  const detectors = [
    {
      id: 'baseline',
      name: 'Baseline',
      status: 'success',
      label: 'REAL',
      probability_ai: 0.12,
      probability_real: 0.88,
    },
    {
      id: 'face_crop',
      name: 'Face-Crop',
      status: 'success',
      label: 'FAKE',
      probability_ai: 0.65,
      probability_real: 0.35,
    },
    {
      id: 'ultra',
      name: 'Ultra',
      status: 'success',
      label: 'REAL',
      probability_ai: 0.08,
      probability_real: 0.92,
    },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'REAL');
  assert.equal(result.label, 'REAL');
  assert.equal(result.majority_reached, true);
  assert.deepEqual(result.vote_counts, { FAKE: 1, REAL: 2, UNCERTAIN: 0 });
  // Verify it took P(REAL) = [0.88, 0.92], NOT P(FAKE)
  assert.deepEqual(result.decision.winning_detector_confidences, [0.88, 0.92]);

  // Expected RMS of [0.88, 0.92] = sqrt((0.7744 + 0.8464) / 2) = sqrt(0.8104) = 0.900222...
  approxEqual(result.confidence, 0.900222, 1e-5);
});

// ---------------------------------------------------------------------------
// 3. Unanimous Votes (3 of 3)
// ---------------------------------------------------------------------------
test('Unanimous FAKE (3 of 3 votes)', () => {
  const detectors = [
    {
      id: 'baseline',
      status: 'success',
      label: 'FAKE',
      probability_ai: 0.85,
      probability_real: 0.15,
    },
    {
      id: 'face_crop',
      status: 'success',
      label: 'FAKE',
      probability_ai: 0.90,
      probability_real: 0.10,
    },
    {
      id: 'ultra',
      status: 'success',
      label: 'FAKE',
      probability_ai: 0.94,
      probability_real: 0.06,
    },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'FAKE');
  assert.equal(result.majority_reached, true);
  assert.deepEqual(result.vote_counts, { FAKE: 3, REAL: 0, UNCERTAIN: 0 });
  assert.deepEqual(result.decision.winning_detector_confidences, [0.85, 0.90, 0.94]);
  approxEqual(result.confidence, 0.897422, 1e-5);
});

test('Unanimous REAL (3 of 3 votes)', () => {
  const detectors = [
    {
      id: 'baseline',
      status: 'success',
      label: 'REAL',
      probability_ai: 0.08,
      probability_real: 0.92,
    },
    {
      id: 'face_crop',
      status: 'success',
      label: 'REAL',
      probability_ai: 0.06,
      probability_real: 0.94,
    },
    {
      id: 'ultra',
      status: 'success',
      label: 'REAL',
      probability_ai: 0.04,
      probability_real: 0.96,
    },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'REAL');
  assert.equal(result.majority_reached, true);
  assert.deepEqual(result.vote_counts, { FAKE: 0, REAL: 3, UNCERTAIN: 0 });
  assert.deepEqual(result.decision.winning_detector_confidences, [0.92, 0.94, 0.96]);
  approxEqual(result.confidence, 0.940142, 1e-5);
});

// ---------------------------------------------------------------------------
// 4. Invalid Detector Outputs & Failures -> UNCERTAIN (0.0)
// ---------------------------------------------------------------------------
test('Failure: Detector 3 Offline (status error)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'FAKE', probability_ai: 0.85, probability_real: 0.15 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.90, probability_real: 0.10 },
    { id: 'ultra', status: 'error', error: 'HTTP 500: Server Error', label: null, probability_ai: null, probability_real: null },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
  assert.ok(result.explanation.includes('UNCERTAIN'));
  assert.ok(result.explanation.includes('failed or returned an incomplete response'));
});

test('Failure: Missing label (null)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: null, probability_ai: 0.85, probability_real: 0.15 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.90, probability_real: 0.10 },
    { id: 'ultra', status: 'success', label: 'FAKE', probability_ai: 0.88, probability_real: 0.12 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
});

test('Failure: Missing label (undefined or unrecognized)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: undefined, probability_ai: 0.85, probability_real: 0.15 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.90, probability_real: 0.10 },
    { id: 'ultra', status: 'success', label: 'FAKE', probability_ai: 0.88, probability_real: 0.12 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
});

test('Failure: Invalid label (UNCERTAIN string)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'UNCERTAIN', probability_ai: 0.50, probability_real: 0.50 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.90, probability_real: 0.10 },
    { id: 'ultra', status: 'success', label: 'FAKE', probability_ai: 0.88, probability_real: 0.12 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
});

test('Failure: NaN probability', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'REAL', probability_ai: NaN, probability_real: 0.90 },
    { id: 'face_crop', status: 'success', label: 'REAL', probability_ai: 0.10, probability_real: 0.90 },
    { id: 'ultra', status: 'success', label: 'REAL', probability_ai: 0.15, probability_real: 0.85 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
  assert.ok(result.explanation.includes('invalid or out-of-bounds probability'));
});

test('Failure: Out-of-bounds probability (> 1.0)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'FAKE', probability_ai: 1.45, probability_real: 0.10 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.80, probability_real: 0.20 },
    { id: 'ultra', status: 'success', label: 'FAKE', probability_ai: 0.85, probability_real: 0.15 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
  assert.ok(result.explanation.includes('invalid or out-of-bounds probability'));
});

test('Failure: Out-of-bounds probability (< 0.0)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'FAKE', probability_ai: -0.15, probability_real: 1.15 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.80, probability_real: 0.20 },
    { id: 'ultra', status: 'success', label: 'FAKE', probability_ai: 0.85, probability_real: 0.15 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
});

test('Failure: Non-numeric probability (string)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'FAKE', probability_ai: '0.90', probability_real: 0.10 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.80, probability_real: 0.20 },
    { id: 'ultra', status: 'success', label: 'FAKE', probability_ai: 0.85, probability_real: 0.15 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
});

test('Failure: Incomplete detector list (only 2 detectors)', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'FAKE', probability_ai: 0.80, probability_real: 0.20 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.85, probability_real: 0.15 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'UNCERTAIN');
  assert.equal(result.confidence, 0.0);
  assert.equal(result.majority_reached, false);
});

// ---------------------------------------------------------------------------
// 5. Adjusted Display Score for Ensemble Confidence (Display-Only Rule)
// ---------------------------------------------------------------------------
test('Adjusted Display Score: Required Examples (Percentage Scale)', () => {
  // - 60% -> 80%
  assert.equal(calculateAdjustedDisplayScore(60), 80);
  // - 55% -> 75%
  assert.equal(calculateAdjustedDisplayScore(55), 75);
  // - 45% -> 65%
  assert.equal(calculateAdjustedDisplayScore(45), 65);
  // - 69% -> 89%
  assert.equal(calculateAdjustedDisplayScore(69), 89);
  // - 70% -> 70% (unchanged)
  assert.equal(calculateAdjustedDisplayScore(70), 70);
  // - 85% -> 85% (unchanged)
  assert.equal(calculateAdjustedDisplayScore(85), 85);
});

test('Adjusted Display Score: Required Examples (Decimal Scale [0.0, 1.0])', () => {
  // - 0.60 -> 0.80
  approxEqual(calculateAdjustedDisplayScore(0.60), 0.80, 1e-6);
  // - 0.55 -> 0.75
  approxEqual(calculateAdjustedDisplayScore(0.55), 0.75, 1e-6);
  // - 0.45 -> 0.65
  approxEqual(calculateAdjustedDisplayScore(0.45), 0.65, 1e-6);
  // - 0.69 -> 0.89
  approxEqual(calculateAdjustedDisplayScore(0.69), 0.89, 1e-6);
  // - 0.70 -> 0.70
  approxEqual(calculateAdjustedDisplayScore(0.70), 0.70, 1e-6);
  // - 0.85 -> 0.85
  approxEqual(calculateAdjustedDisplayScore(0.85), 0.85, 1e-6);
});

test('Adjusted Display Score: String Percentage Inputs', () => {
  assert.equal(calculateAdjustedDisplayScore('60%'), 80);
  assert.equal(calculateAdjustedDisplayScore('55%'), 75);
  assert.equal(calculateAdjustedDisplayScore('45%'), 65);
  assert.equal(calculateAdjustedDisplayScore('69%'), 89);
  assert.equal(calculateAdjustedDisplayScore('70%'), 70);
  assert.equal(calculateAdjustedDisplayScore('85%'), 85);
});

test('Adjusted Display Score: 70% Boundary Verification', () => {
  // Just below 70%: must add 20 percentage points
  assert.equal(calculateAdjustedDisplayScore(69.9), 89.9);
  assert.equal(calculateAdjustedDisplayScore(69.99), 89.99);
  approxEqual(calculateAdjustedDisplayScore(0.699), 0.899, 1e-6);

  // Exactly 70%: boundary condition, must leave unchanged
  assert.equal(calculateAdjustedDisplayScore(70.0), 70.0);
  approxEqual(calculateAdjustedDisplayScore(0.70), 0.70, 1e-6);

  // Just above 70%: must leave unchanged
  assert.equal(calculateAdjustedDisplayScore(70.01), 70.01);
  assert.equal(calculateAdjustedDisplayScore(70.1), 70.1);
  approxEqual(calculateAdjustedDisplayScore(0.701), 0.701, 1e-6);
});

test('Adjusted Display Score: 100% Cap Verification', () => {
  // At exactly 100% (or 1.0)
  assert.equal(calculateAdjustedDisplayScore(100), 100);
  assert.equal(calculateAdjustedDisplayScore(1.0), 1.0);

  // Exceeding 100% capped at 100%
  assert.equal(calculateAdjustedDisplayScore(105), 100);
  assert.equal(calculateAdjustedDisplayScore(120), 100);
  assert.equal(calculateAdjustedDisplayScore(1.05), 1.0);
  assert.equal(calculateAdjustedDisplayScore(1.20), 1.0);
  assert.equal(calculateAdjustedDisplayScore('105%'), 100);
});

test('Adjusted Display Score: String Formatter Helper', () => {
  assert.equal(formatAdjustedDisplayScore(0.60), '80.0%');
  assert.equal(formatAdjustedDisplayScore(0.55), '75.0%');
  assert.equal(formatAdjustedDisplayScore(0.45), '65.0%');
  assert.equal(formatAdjustedDisplayScore(0.69), '89.0%');
  assert.equal(formatAdjustedDisplayScore(0.70), '70.0%');
  assert.equal(formatAdjustedDisplayScore(0.85), '85.0%');
  assert.equal(formatAdjustedDisplayScore('60%'), '80.0%');
  assert.equal(formatAdjustedDisplayScore('70%'), '70.0%');
  assert.equal(formatAdjustedDisplayScore(null), 'Inconclusive');
  assert.equal(formatAdjustedDisplayScore(undefined), 'Inconclusive');
});

test('Adjusted Display Score: Isolation & Non-interference with Ensemble Logic', () => {
  const detectors = [
    { id: 'baseline', status: 'success', label: 'FAKE', probability_ai: 0.60, probability_real: 0.40 },
    { id: 'face_crop', status: 'success', label: 'FAKE', probability_ai: 0.60, probability_real: 0.40 },
    { id: 'ultra', status: 'success', label: 'REAL', probability_ai: 0.20, probability_real: 0.80 },
  ];

  const result = aggregateEnsembleVoting(detectors);
  assert.equal(result.prediction, 'FAKE');
  // Raw RMS of [0.60, 0.60] = 0.60
  approxEqual(result.confidence, 0.60, 1e-5);

  // Calculate display score separately
  const displayScore = calculateAdjustedDisplayScore(result.confidence);
  approxEqual(displayScore, 0.80, 1e-5);

  // Verify that computing display score did not mutate or alter original ensemble result
  approxEqual(result.confidence, 0.60, 1e-5, 'Raw RMS confidence preserved');
  assert.equal(result.prediction, 'FAKE', 'Classification preserved');
  assert.deepEqual(result.vote_counts, { FAKE: 2, REAL: 1, UNCERTAIN: 0 }, 'Votes preserved');
  assert.equal(result.decision.ensemble_confidence_score, 0.60, 'Ensemble decision score preserved');
  assert.equal(detectors[0].probability_ai, 0.60, 'Detector raw output preserved');
});

// ---------------------------------------------------------------------------
// 6. Startup Health Checks & Non-blocking Readiness
// ---------------------------------------------------------------------------
test('Startup Health Check: Correctly aggregates when both backends are ready', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async (url) => {
      if (String(url).includes('8000')) {
        return { ok: true, status: 200, json: async () => ({ model_name: 'efficientnet_b0', weights_loaded: true }) };
      }
      if (String(url).includes('8005')) {
        return { ok: true, status: 200, json: async () => ({ status: 'ok', model: { backend: 'ultra' } }) };
      }
      return { ok: false, status: 404 };
    };

    const health = await checkServicesHealth({ timeoutMs: 1000 });
    assert.equal(health.allReady, true);
    assert.equal(health.deepguard.ready, true);
    assert.equal(health.ultra.ready, true);
    assert.equal(health.detectorsReadyCount, 3);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Startup Health Check: Handles partial detector availability gracefully without crashing', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async (url) => {
      if (String(url).includes('8000')) {
        return { ok: true, status: 200, json: async () => ({ model_name: 'efficientnet_b0', weights_loaded: true }) };
      }
      if (String(url).includes('8005')) {
        throw new Error('ECONNREFUSED: Connection refused at 127.0.0.1:8005');
      }
      return { ok: false, status: 500 };
    };

    const health = await checkServicesHealth({ timeoutMs: 1000 });
    assert.equal(health.allReady, false);
    assert.equal(health.deepguard.ready, true);
    assert.equal(health.ultra.ready, false);
    assert.equal(health.detectorsReadyCount, 2); // 2 out of 3 detectors available
    assert.ok(health.ultra.error);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('Startup Health Check: Times out properly when endpoint hangs', async () => {
  const originalFetch = globalThis.fetch;
  try {
    // Hangs indefinitely until AbortSignal fires
    globalThis.fetch = (url, options) => {
      return new Promise((resolve, reject) => {
        if (options?.signal) {
          options.signal.addEventListener('abort', () => {
            const err = new Error('The operation was aborted');
            err.name = 'AbortError';
            reject(err);
          });
        }
      });
    };

    const startTime = Date.now();
    await assert.rejects(
      async () => {
        await checkBackendStatus(100); // 100ms timeout
      },
      (err) => {
        assert.equal(err.name, 'Error');
        assert.ok(err.message.includes('timed out'));
        return true;
      }
    );
    const duration = Date.now() - startTime;
    assert.ok(duration < 500, `Expected timeout around 100ms, completed in ${duration}ms`);
  } finally {
    globalThis.fetch = originalFetch;
  }
});


