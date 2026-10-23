import test from 'node:test';
import assert from 'node:assert/strict';
import { predictFile } from '../src/services/predictionApi.js';

test('predictFile: rejects video inference with informative error', async () => {
  await assert.rejects(
    async () => {
      await predictFile(new File(['fake-video'], 'video.mp4', { type: 'video/mp4' }), true);
    },
    {
      name: 'Error',
      message: /Video inference is not supported by the verified AI Image Detector backend/
    }
  );
});

test('predictFile: rejects invalid or null file payload', async () => {
  await assert.rejects(
    async () => {
      await predictFile(null, false);
    },
    {
      name: 'Error',
      message: /Invalid file payload/
    }
  );
});

test('predictFile: parses authentic REAL DetectionResult from Project A backend without altering values', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => ({
      ok: true,
      json: async () => ({
        path: null,
        label: 'real',
        probability_ai: 0.3064,
        probability_real: 0.6936,
        confidence: 0.6936,
        raw_score: -0.8168,
        backend: 'univfd',
        filename: 'human.jpeg'
      })
    });

    const dummyFile = new File(['dummy-bytes'], 'human.jpeg', { type: 'image/jpeg' });
    const result = await predictFile(dummyFile, false);

    assert.equal(result.prediction, 'REAL');
    assert.equal(result.label, 'REAL');
    assert.equal(result.probability_ai, 0.3064);
    assert.equal(result.probability_real, 0.6936);
    assert.equal(result.confidence, 0.6936);
    assert.equal(result.raw_score, -0.8168);
    assert.equal(result.backend, 'univfd');
    assert.equal(result.filename, 'human.jpeg');
    assert.equal(result.decision.operating_mode, 'AI Image Detector (UNIVFD)');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('predictFile: parses authentic AI DetectionResult from Project A backend without altering values', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => ({
      ok: true,
      json: async () => ({
        path: null,
        label: 'ai',
        probability_ai: 0.885,
        probability_real: 0.115,
        confidence: 0.885,
        raw_score: 2.041,
        backend: 'univfd',
        filename: 'synthetic.png'
      })
    });

    const dummyFile = new File(['dummy-bytes'], 'synthetic.png', { type: 'image/png' });
    const result = await predictFile(dummyFile, false);

    assert.equal(result.prediction, 'AI GENERATED');
    assert.equal(result.label, 'AI');
    assert.equal(result.probability_ai, 0.885);
    assert.equal(result.probability_real, 0.115);
    assert.equal(result.confidence, 0.885);
    assert.equal(result.raw_score, 2.041);
    assert.equal(result.backend, 'univfd');
    assert.equal(result.filename, 'synthetic.png');
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('predictFile: handles backend HTTP errors gracefully', async () => {
  const originalFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => ({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ detail: 'Corrupt image bytes' })
    });

    const dummyFile = new File(['bad-bytes'], 'corrupt.jpg', { type: 'image/jpeg' });
    await assert.rejects(
      async () => {
        await predictFile(dummyFile, false);
      },
      {
        name: 'Error',
        message: /Invalid File: Corrupt image bytes/
      }
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
