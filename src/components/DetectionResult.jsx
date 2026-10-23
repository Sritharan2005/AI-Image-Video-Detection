import React, { useEffect } from 'react';
import {
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Info,
  Film,
  Clock,
} from 'lucide-react';
import { calculateAdjustedDisplayScore } from './displayScore.js';

export default function DetectionResult({
  result,
  previewUrl,
  isVideo,
  onAnalyzeAnother,
}) {
  if (!result) return null;

  const isVideoResult = isVideo || result.media_type === 'video' || !!result.per_frame_predictions;
  const isReal = result.label ? result.label.toLowerCase() === 'real' : result.prediction === 'REAL';
  const isFake = result.label
    ? (result.label.toLowerCase() === 'fake' || result.label.toLowerCase() === 'ai')
    : result.prediction === 'FAKE';
  const isUncertain = result.prediction === 'UNCERTAIN' || (!isReal && !isFake);

  const rawConfidencePercent = isUncertain
    ? 'Inconclusive'
    : `${(result.confidence * 100).toFixed(1)}%`;

  // Calculate separate adjusted display score strictly for display presentation
  // (Original score < 70% -> +20%, >= 70% -> unchanged, capped at 100%)
  const adjustedDisplayScore = calculateAdjustedDisplayScore(result.confidence);
  const adjustedDisplayPercent = isUncertain
    ? 'Inconclusive'
    : `${(adjustedDisplayScore * 100).toFixed(1)}%`;

  const displayedScorePercent = isVideoResult ? rawConfidencePercent : adjustedDisplayPercent;
  const displayedScoreWidth = isUncertain
    ? '0%'
    : (isVideoResult
        ? `${(result.confidence * 100).toFixed(1)}%`
        : `${(adjustedDisplayScore * 100).toFixed(1)}%`);

  // Diagnostic logging to Developer Tools (keeps detailed signals accessible without cluttering the user interface)
  useEffect(() => {
    if (result) {
      console.group('DeepGuard AI Detection — Internal Diagnostics');
      console.info('Classification:', result.prediction);
      console.info('Ensemble Confidence (RMS):', result.confidence);
      if (!isVideoResult) {
        console.info('Adjusted Display Score:', adjustedDisplayScore);
      }
      if (result.individual_detectors) {
        console.table(result.individual_detectors.map(d => ({
          Detector: d.name || d.id,
          Port: d.port,
          Status: d.status,
          Vote: d.label,
          'P(AI)': typeof d.probability_ai === 'number' ? `${(d.probability_ai * 100).toFixed(2)}%` : 'N/A',
          'P(REAL)': typeof d.probability_real === 'number' ? `${(d.probability_real * 100).toFixed(2)}%` : 'N/A',
          Confidence: typeof d.confidence === 'number' ? `${(d.confidence * 100).toFixed(2)}%` : 'N/A',
          Error: d.error || 'None'
        })));
      }
      if (result.decision) {
        console.dir(result.decision);
      }
      console.groupEnd();
    }
  }, [result, adjustedDisplayScore, isVideoResult]);

  // Clean final classification label: 'AI-Generated (FAKE)' or 'Likely Real'
  let displayTitle = 'Detection Unavailable';
  if (!isUncertain) {
    displayTitle = isFake ? 'AI-Generated (FAKE)' : 'Likely Real';
  }

  // Concise explanation that the result is based on multiple detector signals
  let explanationText = 'Detection temporarily unavailable. Please try again.';
  if (!isUncertain) {
    if (isVideoResult) {
      explanationText = isFake
        ? 'This video was classified as AI-Generated (FAKE) based on spatial feature evaluations across sampled frames.'
        : 'This video was classified as Likely Real based on spatial feature evaluations across sampled frames.';
    } else {
      explanationText = isFake
        ? 'This image was classified as AI-Generated (FAKE) based on consensus across multiple independent detection signals identifying synthetic patterns.'
        : 'This image was classified as Likely Real based on consensus across multiple independent detection signals consistent with authentic photography.';
    }
  }

  return (
    <section id="detection-result" className="result-section">
      <div className="container">
        <div className="result-card">
          {/* Header */}
          <div className="result-header">
            <div>
              <h2 className="result-title">
                {isVideoResult ? 'Video Analysis Result' : 'Detection Result'}
              </h2>
              <div style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '2px' }}>
                {isVideoResult
                  ? 'Sampled frame-level spatial evaluation across video duration'
                  : 'Multi-signal neural network evaluation'}
              </div>
            </div>
            <button
              type="button"
              className="btn-analyze-another"
              onClick={onAnalyzeAnother}
            >
              <RefreshCw size={15} />
              <span>Analyze Another File</span>
            </button>
          </div>

          {/* Unified Result Grid */}
          <div className="result-grid">
            {/* Left Column: Media Thumbnail Preview */}
            <div className="result-preview-box">
              {isVideoResult ? (
                <video
                  src={previewUrl}
                  controls
                  className="result-preview-img"
                  style={{ maxHeight: '280px', objectFit: 'contain', backgroundColor: '#000000' }}
                />
              ) : (
                <img
                  src={previewUrl}
                  alt="Analyzed media preview"
                  className="result-preview-img"
                />
              )}
              {isVideoResult && (
                <div style={{
                  position: 'absolute',
                  top: '8px',
                  left: '8px',
                  backgroundColor: 'rgba(16, 24, 40, 0.85)',
                  color: '#FFFFFF',
                  fontSize: '11px',
                  padding: '3px 8px',
                  borderRadius: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  <Film size={12} />
                  <span>{result.video_metadata?.duration_seconds ?? 0}s • {result.video_metadata?.frames_sampled ?? 0} frames</span>
                </div>
              )}
            </div>

            {/* Center Column: Status Badge & Concise Explanation */}
            <div className="result-status-column">
              <div
                className={`result-status-badge ${isUncertain ? 'uncertain' : (isReal ? 'real' : 'fake')}`}
                style={isUncertain ? {
                  backgroundColor: 'rgba(245, 158, 11, 0.1)',
                  borderColor: 'rgba(245, 158, 11, 0.4)',
                  color: '#B45309'
                } : {}}
              >
                <div className="status-left">
                  {isUncertain ? (
                    <HelpCircle className="status-icon" strokeWidth={2.4} color="#D97706" />
                  ) : isReal ? (
                    <CheckCircle2 className="status-icon" strokeWidth={2.4} />
                  ) : (
                    <AlertTriangle className="status-icon" strokeWidth={2.4} />
                  )}
                  <span className="status-label" style={isUncertain ? { color: '#B45309', fontSize: '15px' } : {}}>
                    {displayTitle}
                  </span>
                </div>

                <div className="status-confidence-group">
                  <div className="confidence-title">
                    {isVideoResult ? 'Confidence' : 'Adjusted Display Score'}
                  </div>
                  <div className="confidence-value">
                    {displayedScorePercent}
                  </div>
                </div>
              </div>

              <p className="result-explanation">
                {explanationText}
              </p>

              {/* Disclaimer Notice */}
              <div style={{
                marginTop: '16px',
                padding: '10px 14px',
                background: 'var(--bg-subtle)',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border-light)',
                fontSize: '12px',
                color: 'var(--text-secondary)',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '10px',
                lineHeight: '1.45'
              }}>
                <Info size={16} color="var(--primary)" style={{ flexShrink: 0, marginTop: '2px' }} />
                <span>
                  <strong>Disclaimer:</strong> Ensemble confidence scores reflect agreement across model signals and do not represent a calibrated probability of correctness. Detection is probabilistic; evaluate outputs alongside contextual metadata.
                </span>
              </div>
            </div>

            {/* Right Column: Clean Decision Summary Card */}
            <div className="result-analysis-column">
              <h3 className="analysis-heading">
                Detection Summary
              </h3>

              <div className="metrics-list" style={{ marginTop: '16px' }}>
                <div className="metric-row">
                  <div className="metric-meta">
                    <span className="metric-name" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {isVideoResult ? 'Confidence Score' : 'Adjusted Display Score'}
                    </span>
                    <span
                      className="metric-score"
                      style={{
                        color: isReal ? 'var(--real-green)' : (isUncertain ? '#B45309' : 'var(--fake-red)'),
                        fontWeight: 800,
                        fontSize: '16px'
                      }}
                    >
                      {displayedScorePercent}
                    </span>
                  </div>
                  <div className="metric-bar-bg" style={{ height: '8px' }}>
                    <div
                      className="metric-bar-fill"
                      style={{
                        width: displayedScoreWidth,
                        backgroundColor: isReal ? 'var(--real-green)' : (isUncertain ? '#F59E0B' : 'var(--fake-red)'),
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Metadata Attributes */}
              <div style={{
                marginTop: '20px',
                padding: '14px 16px',
                background: 'var(--bg-subtle)',
                border: '1px solid var(--border-light)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '13px',
                lineHeight: '1.7',
                color: 'var(--text-secondary)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span>Analysis Mode:</span>
                  <strong style={{ color: 'var(--text-primary)' }}>
                    {isVideoResult ? 'DeepGuard Video Analysis' : 'DeepGuard AI Detection'}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span>Evaluation:</span>
                  <strong style={{ color: 'var(--text-primary)' }}>
                    {isVideoResult ? 'Sampled Frame Evaluation' : 'Multi-Model Consensus'}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span>Classification:</span>
                  <strong style={{ color: isReal ? 'var(--real-green)' : (isUncertain ? '#B45309' : 'var(--fake-red)') }}>
                    {displayTitle}
                  </strong>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Status:</span>
                  <span style={{
                    color: isUncertain ? '#B45309' : '#10B981',
                    fontWeight: 600,
                  }}>
                    {isUncertain ? 'Unavailable' : 'Verified'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Video Per-Frame Breakdown Section (Only shown for video files) */}
          {isVideoResult && result.per_frame_predictions && result.per_frame_predictions.length > 0 && (
            <div style={{ marginTop: '28px', borderTop: '1px solid var(--border-light)', paddingTop: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px', marginBottom: '14px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Film size={18} color="var(--primary)" />
                  <h3 style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', margin: 0 }}>
                    Timestamped Per-Frame Analysis ({result.per_frame_predictions.length} Frames Evaluated)
                  </h3>
                </div>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)', background: 'var(--bg-subtle)', padding: '3px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-light)' }}>
                  Aggregation: {result.decision?.aggregation_method?.toUpperCase() || 'MEAN'}
                </span>
              </div>

              {/* Frame Cards Grid */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))',
                gap: '12px',
              }}>
                {result.per_frame_predictions.map((frame, idx) => {
                  const frameIsReal = frame.prediction === 'REAL';
                  const frameFakeScore = frame.probabilities?.FAKE ?? frame.raw_probabilities?.FAKE ?? 0.0;
                  const frameFakePct = (frameFakeScore * 100).toFixed(1);

                  return (
                    <div
                      key={idx}
                      style={{
                        background: 'var(--bg-subtle)',
                        border: '1px solid var(--border-light)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '12px 14px',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '8px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                          <Clock size={13} />
                          <span>{frame.timestamp_display || `${frame.timestamp_seconds.toFixed(2)}s`}</span>
                        </div>
                        <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          Frame #{frame.frame_index}
                        </span>
                      </div>

                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        background: frameIsReal ? 'var(--real-green-bg)' : 'var(--fake-red-bg)',
                        border: `1px solid ${frameIsReal ? 'var(--real-green-border)' : 'var(--fake-red-border)'}`,
                      }}>
                        <span style={{
                          fontSize: '12px',
                          fontWeight: 700,
                          color: frameIsReal ? 'var(--real-green)' : 'var(--fake-red)',
                        }}>
                          {frame.prediction}
                        </span>
                        <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                          {(frame.confidence * 100).toFixed(1)}% conf
                        </span>
                      </div>

                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '3px' }}>
                          <span>P(FAKE)</span>
                          <span style={{ fontWeight: 600, color: frameFakeScore >= 0.40 ? 'var(--fake-red)' : 'var(--text-secondary)' }}>
                            {frameFakePct}%
                          </span>
                        </div>
                        <div style={{ height: '4px', background: 'var(--border-light)', borderRadius: '2px', overflow: 'hidden' }}>
                          <div
                            style={{
                              width: `${frameFakePct}%`,
                              height: '100%',
                              backgroundColor: frameFakeScore >= 0.40 ? 'var(--fake-red)' : 'var(--real-green)',
                              borderRadius: '2px',
                            }}
                          />
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
