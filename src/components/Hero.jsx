import React from 'react';
import { ArrowRight, Sparkles } from 'lucide-react';
import ComparisonSlider from './ComparisonSlider';

export default function Hero({ onTryDetection, onLearnMore }) {
  return (
    <section id="home" className="hero-section">
      <div className="container">
        <div className="hero-grid">
          {/* Left Column */}
          <div className="hero-text-col">
            <div className="hero-badge">
              <Sparkles size={13} strokeWidth={2.5} />
              <span>AI POWERED</span>
            </div>

            <h1 className="hero-title">
              Detect <span className="highlight">Fake Images</span>
              <br />
              and Videos Easily
            </h1>

            <p className="hero-desc">
              Upload an image or video and let our AI analyze it for manipulated or AI-generated content. Get accurate results with confidence score and visual analysis.
            </p>

            <div className="hero-cta-group">
              <button
                type="button"
                className="btn-primary"
                onClick={onTryDetection}
              >
                Try Detection
                <ArrowRight size={16} />
              </button>

              <button
                type="button"
                className="btn-secondary"
                onClick={onLearnMore}
              >
                Learn More
              </button>
            </div>
          </div>

          {/* Right Column: Split Comparison Visual */}
          <div className="hero-visual-col">
            <ComparisonSlider />
          </div>
        </div>
      </div>
    </section>
  );
}
