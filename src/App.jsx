import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import UploadSection from './components/UploadSection';
import DetectionResult from './components/DetectionResult';
import SampleGallery from './components/SampleGallery';
import HowItWorks from './components/HowItWorks';
import Footer from './components/Footer';
import { predictFile, checkServicesHealth } from './services/predictionApi';

export default function App() {
  const [theme, setTheme] = useState('light');
  const [activeSection, setActiveSection] = useState('home');
  const [activeTab, setActiveTab] = useState('image'); // 'image' | 'video'

  // Selected file, preview URL, and inference mode (defaults to DeepGuard baseline on port 8000)
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('/images/sample-real-man.jpg');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [inferenceMode, setInferenceMode] = useState('ensemble_voting'); // 'ensemble_voting' (Default 3-Detector Voting) | 'baseline' (Port 8000) | 'experimental_face' (Port 8000) | 'ultra_detector' (Optional, Port 8005)
  const [apiError, setApiError] = useState(null);

  // Live detection result from API
  const [detectionResult, setDetectionResult] = useState(null);

  // Backend readiness state (runs completely asynchronously, never blocks initial UI render)
  const [backendHealth, setBackendHealth] = useState({
    checked: false,
    checking: true,
    allReady: false,
    detectorsReadyCount: 0,
    totalDetectors: 3,
    statusText: 'Connecting to AI detectors...',
    deepguard: { ready: false },
    ultra: { ready: false },
  });

  useEffect(() => {
    let isMounted = true;
    let retryCount = 0;
    const maxRetries = 3;
    let retryTimer = null;

    const performHealthCheck = async () => {
      try {
        const health = await checkServicesHealth({ timeoutMs: 3000 });
        if (!isMounted) return;

        setBackendHealth({
          checked: true,
          checking: false,
          allReady: health.allReady,
          detectorsReadyCount: health.detectorsReadyCount,
          totalDetectors: 3,
          statusText: health.allReady
            ? 'All AI detectors online & ready'
            : `${health.detectorsReadyCount}/3 AI detectors online (${!health.deepguard.ready ? 'Port 8000 starting' : 'Port 8005 starting'})`,
          deepguard: health.deepguard,
          ultra: health.ultra,
        });

        if (!health.allReady && retryCount < maxRetries) {
          retryCount += 1;
          retryTimer = setTimeout(performHealthCheck, 4000);
        }
      } catch (_) {
        if (!isMounted) return;
        setBackendHealth((prev) => ({
          ...prev,
          checked: true,
          checking: false,
          allReady: false,
          statusText: 'AI detectors initializing in background',
        }));
        if (retryCount < maxRetries) {
          retryCount += 1;
          retryTimer = setTimeout(performHealthCheck, 4000);
        }
      }
    };

    // Perform check asynchronously without blocking UI
    performHealthCheck();

    return () => {
      isMounted = false;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, []);

  // Toggle Theme
  const handleToggleTheme = () => {
    const nextTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(nextTheme);
    document.documentElement.setAttribute('data-theme', nextTheme);
  };

  // Scroll to section
  const handleNavigate = (id) => {
    setActiveSection(id);
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Handle live file analysis via backend
  const handleAnalyzeFile = async (file, isVideo = false) => {
    setIsAnalyzing(true);
    setApiError(null);
    try {
      const mode = isVideo ? 'baseline' : 'ensemble_voting';
      const res = await predictFile(file, isVideo, mode);
      setDetectionResult(res);
      setTimeout(() => {
        document.getElementById('detection-result')?.scrollIntoView({ behavior: 'smooth' });
      }, 150);
    } catch (err) {
      console.error('Analysis error:', err);
      setApiError(err.message || 'Detection temporarily unavailable. Please try again.');
      document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle selecting a sample card from the gallery
  const handleSelectSample = async (sample) => {
    setActiveTab('image');
    setPreviewUrl(sample.imageUrl);
    setSelectedFile({
      name: `${sample.title.toLowerCase().replace(/\s+/g, '-')}.jpg`,
      size: 730864,
      type: 'image/jpeg',
    });
    setApiError(null);
    setIsAnalyzing(true);
    document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' });

    try {
      const res = await predictFile(sample.imageUrl, false, 'ensemble_voting');
      setDetectionResult(res);
      setTimeout(() => {
        document.getElementById('detection-result')?.scrollIntoView({ behavior: 'smooth' });
      }, 200);
    } catch (err) {
      console.error('Sample analysis error:', err);
      setApiError(err.message || 'Detection temporarily unavailable. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle reset to analyze another file
  const handleAnalyzeAnother = () => {
    setSelectedFile(null);
    setDetectionResult(null);
    setApiError(null);
    document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className="app-root">
      {/* Navbar */}
      <Navbar
        theme={theme}
        onToggleTheme={handleToggleTheme}
        activeSection={activeSection}
        onNavigate={handleNavigate}
      />

      <main>
        {/* Hero Section */}
        <Hero
          onTryDetection={() => handleNavigate('upload-section')}
          onLearnMore={() => handleNavigate('how-it-works')}
        />

        {/* Upload Section */}
        <UploadSection
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          onFileAnalyzed={handleAnalyzeFile}
          isAnalyzing={isAnalyzing}
          selectedFile={selectedFile}
          setSelectedFile={setSelectedFile}
          previewUrl={previewUrl}
          setPreviewUrl={setPreviewUrl}
          inferenceMode={inferenceMode}
          setInferenceMode={setInferenceMode}
          apiError={apiError}
          setApiError={setApiError}
          backendHealth={backendHealth}
        />

        {/* Detection Result Section */}
        {detectionResult && (
          <DetectionResult
            result={detectionResult}
            previewUrl={previewUrl}
            isVideo={activeTab === 'video'}
            onAnalyzeAnother={handleAnalyzeAnother}
          />
        )}

        {/* Sample Files Gallery */}
        <SampleGallery onSelectSample={handleSelectSample} />

        {/* How It Works Section */}
        <HowItWorks />
      </main>

      {/* Footer */}
      <Footer onNavigate={handleNavigate} />
    </div>
  );
}