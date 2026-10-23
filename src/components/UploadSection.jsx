import React, { useState, useRef } from 'react';
import {
  Image as ImageIcon,
  Video,
  UploadCloud,
  AlertCircle,
  Trash2,
  ArrowRight,
  Sliders,
  Sparkles,
  Layers,
  Cpu,
  Info,
  ShieldCheck,
  Film,
} from 'lucide-react';

export default function UploadSection({
  activeTab,
  setActiveTab,
  onFileAnalyzed,
  isAnalyzing,
  selectedFile,
  setSelectedFile,
  previewUrl,
  setPreviewUrl,
  inferenceMode,
  setInferenceMode,
  apiError,
  setApiError,
  backendHealth,
}) {
  const [dragOver, setDragOver] = useState(false);
  const [analyzingStep, setAnalyzingStep] = useState('Submitting to neural network...');
  const fileInputRef = useRef(null);

  const isVideo = activeTab === 'video';

  const acceptedImageTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
  const maxImageSizeBytes = 25 * 1024 * 1024; // 25MB

  const acceptedVideoTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska'];
  const maxVideoSizeBytes = 100 * 1024 * 1024; // 100MB

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setApiError(null);
    if (tab === 'video' && selectedFile && !selectedFile.type.startsWith('video/') && !selectedFile.name.match(/\.(mp4|mov|avi|webm|mkv)$/i)) {
      handleRemoveFile();
    } else if (tab === 'image' && selectedFile && !selectedFile.type.startsWith('image/') && !selectedFile.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
      handleRemoveFile();
    }
  };

  const validateAndSetFile = (file) => {
    setApiError(null);

    if (isVideo) {
      const isVideoType = acceptedVideoTypes.includes(file.type) || file.name.match(/\.(mp4|mov|avi|webm|mkv)$/i);
      if (!isVideoType) {
        setApiError('Unsupported format. Please upload a standard video file (MP4, MOV, AVI, WebM).');
        return;
      }

      if (file.size > maxVideoSizeBytes) {
        setApiError('File size exceeds the 100MB limit for videos.');
        return;
      }

      if (file.size === 0) {
        setApiError('The selected file is empty (0 bytes). Please upload a valid video.');
        return;
      }

      setSelectedFile(file);
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);
      return;
    }

    if (!acceptedImageTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
      setApiError('Unsupported format. Please upload a standard image file (JPG, PNG, or WebP).');
      return;
    }

    if (file.size > maxImageSizeBytes) {
      setApiError('File size exceeds the 25MB limit for images.');
      return;
    }

    if (file.size === 0) {
      setApiError('The selected file is empty (0 bytes). Please upload a valid image.');
      return;
    }

    setSelectedFile(file);
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (previewUrl && previewUrl.startsWith('blob:')) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setApiError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyzeClick = () => {
    if (!selectedFile) return;

    if (isVideo) {
      setAnalyzingStep('Decoding video & extracting sampled frames...');
      onFileAnalyzed(selectedFile, true, inferenceMode);
      return;
    }

    setAnalyzingStep('Running DeepGuard AI Detection across multi-model signals...');
    onFileAnalyzed(selectedFile, false, 'ensemble_voting');
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
  };

  return (
    <section id="upload-section" className="upload-section">
      <div className="container">
        <div className="upload-card">
          {/* Top Tabs */}
          <div className="upload-header-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '20px' }}>
            <div className="upload-tabs" style={{ marginBottom: 0 }}>
              <button
                type="button"
                className={`upload-tab ${activeTab === 'image' ? 'active' : ''}`}
                onClick={() => handleTabChange('image')}
              >
                <ImageIcon size={18} />
                <span>Image Analysis</span>
              </button>
              <button
                type="button"
                className={`upload-tab ${activeTab === 'video' ? 'active' : ''}`}
                onClick={() => handleTabChange('video')}
              >
                <Video size={18} />
                <span>Video Analysis</span>
              </button>
            </div>

            {/* Unified Detection Mode Indicator */}
            <div className="mode-indicator" style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              background: 'var(--bg-subtle)',
              padding: '6px 14px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-light)',
              fontSize: '13px',
              fontWeight: '600',
              color: 'var(--text-primary)',
            }}>
              {!isVideo ? (
                <>
                  <span style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    backgroundColor: backendHealth?.allReady
                      ? '#10B981'
                      : (backendHealth?.checking ? '#F59E0B' : (backendHealth?.detectorsReadyCount > 0 ? '#F59E0B' : '#EF4444')),
                    display: 'inline-block',
                  }} />
                  <ShieldCheck size={16} color="var(--primary)" />
                  <span>
                    DeepGuard AI Detection
                    {backendHealth?.checked && (
                      <span style={{ fontWeight: 500, fontSize: '12px', color: 'var(--text-secondary)', marginLeft: '6px' }}>
                        ({backendHealth.allReady ? '3/3 Detectors Online' : `${backendHealth.detectorsReadyCount}/3 Online`})
                      </span>
                    )}
                  </span>
                </>
              ) : (
                <>
                  <Film size={16} color="var(--primary)" />
                  <span>DeepGuard Video Analysis</span>
                </>
              )}
            </div>
          </div>

          {/* Subtle Detector Readiness Status Notice when starting/initializing */}
          {!isVideo && backendHealth?.checked && !backendHealth?.allReady && (
            <div style={{
              background: 'rgba(245, 158, 11, 0.08)',
              border: '1px solid rgba(245, 158, 11, 0.25)',
              borderRadius: 'var(--radius-md)',
              padding: '10px 16px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              fontSize: '12px',
              color: '#B45309',
              lineHeight: '1.4'
            }}>
              <AlertCircle size={16} style={{ flexShrink: 0 }} />
              <span>
                <strong>Detector Status:</strong> AI backend services are initializing ({backendHealth.statusText || 'models loading in background'}). You can select files now; full voting analysis will evaluate once models finish loading.
              </span>
            </div>
          )}

          {/* Image Realistic AI Disclaimer Banner */}
          {!isVideo && (
            <div style={{
              background: 'var(--bg-subtle)',
              border: '1px solid var(--border-light)',
              borderRadius: 'var(--radius-md)',
              padding: '12px 18px',
              marginBottom: '20px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              fontSize: '13px',
              color: 'var(--text-secondary)'
            }}>
              <Info size={18} color="var(--primary)" style={{ flexShrink: 0 }} />
              <span>
                <strong>Detection Notice:</strong> Highly realistic AI-generated images and subtle diffusion renders may lack detectable synthetic artifacts. DeepGuard AI Detection evaluates multiple complementary feature signals to maximize detection reliability.
              </span>
            </div>
          )}

          {/* Video Notice Banner */}
          {isVideo && (
            <div style={{ background: 'var(--primary-light)', border: '1px solid rgba(13, 91, 255, 0.2)', borderRadius: 'var(--radius-md)', padding: '14px 18px', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <Info size={20} color="var(--primary)" style={{ flexShrink: 0 }} />
              <span style={{ fontSize: '13px', color: 'var(--text-primary)' }}>
                <strong>Video Frame Analysis:</strong> Samples evenly distributed frames across the video and evaluates static spatial features with EfficientNet-B0. (Does not measure temporal continuity or video compression dynamics).
              </span>
            </div>
          )}

          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            accept={isVideo ? 'video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.mov,.avi,.webm,.mkv' : 'image/jpeg,image/png,image/jpg,image/webp'}
            onChange={handleFileInputChange}
            style={{ display: 'none' }}
          />

          {/* File Selected State */}
          {selectedFile && !isAnalyzing ? (
            <div>
              <div className="file-selected-box">
                <div className="file-info-group">
                  {isVideo ? (
                    <video
                      src={previewUrl}
                      className="file-thumb-preview"
                      muted
                      preload="metadata"
                      style={{ objectFit: 'cover' }}
                    />
                  ) : (
                    <img
                      src={previewUrl}
                      alt="Selected upload preview"
                      className="file-thumb-preview"
                    />
                  )}
                  <div>
                    <div className="file-name" title={selectedFile.name}>
                      {selectedFile.name}
                    </div>
                    <div className="file-size">
                      {formatFileSize(selectedFile.size)} • Mode: {isVideo ? 'DeepGuard Video Analysis' : 'DeepGuard AI Detection'}
                    </div>
                  </div>
                </div>

                <div className="file-actions-group">
                  <button
                    type="button"
                    className="btn-remove-file"
                    onClick={handleRemoveFile}
                  >
                    <Trash2 size={16} />
                    <span>Remove</span>
                  </button>

                  <button
                    type="button"
                    className="btn-primary"
                    onClick={handleAnalyzeClick}
                  >
                    <span>{isVideo ? 'Analyze Video' : 'Analyze Image'}</span>
                    <ArrowRight size={16} />
                  </button>
                </div>
              </div>
            </div>
          ) : isAnalyzing ? (
            /* Analyzing / Loading State */
            <div className="analyzing-container">
              <div className="analyzing-spinner" />
              <div className="analyzing-step-text">{analyzingStep}</div>
              <div className="analyzing-subtext">
                {isVideo
                  ? 'Extracting video frames and running spatial evaluation across sampled frames...'
                  : 'Evaluating multi-model neural network signals across spatial and frequency domains...'}
              </div>
            </div>
          ) : (
            /* Upload Drop Area */
            <div
              className={`upload-dropzone ${dragOver ? 'dragging' : ''}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="upload-icon-wrapper">
                <UploadCloud size={46} strokeWidth={2.2} />
              </div>

              <div className="upload-title">
                {isVideo ? 'Drag & drop a video file here' : 'Drag & drop an image here'}
              </div>

              <div className="upload-or">or</div>

              <button
                type="button"
                className="btn-primary upload-btn-choose"
                onClick={(e) => {
                  e.stopPropagation();
                  fileInputRef.current?.click();
                }}
              >
                {isVideo ? 'Choose Video File' : 'Choose Image File'}
              </button>

              <div className="upload-hint">
                {isVideo
                  ? 'Supported formats: MP4, MOV, AVI, WebM (Max 100MB)'
                  : 'Supported formats: JPG, PNG, JPEG, WebP (Max 25MB)'}
              </div>
            </div>
          )}

          {/* Error Message Banner */}
          {apiError && (
            <div className="error-banner" style={{ marginTop: '16px' }}>
              <AlertCircle size={18} style={{ flexShrink: 0 }} />
              <div style={{ flex: 1 }}>{apiError}</div>
              <button
                type="button"
                onClick={() => setApiError(null)}
                style={{ fontSize: '12px', fontWeight: 'bold', color: 'inherit', textDecoration: 'underline', marginLeft: '12px' }}
              >
                Dismiss
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
