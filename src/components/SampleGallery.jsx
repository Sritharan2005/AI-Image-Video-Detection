import React from 'react';
import { ArrowRight } from 'lucide-react';
import { sampleFiles } from '../data/samples';

export default function SampleGallery({ onSelectSample }) {
  return (
    <section id="sample-gallery" className="samples-section">
      <div className="container">
        {/* Header */}
        <div className="samples-header">
          <h2 className="samples-title">Try with Sample Files</h2>
          <a
            href="#sample-gallery"
            className="samples-link-more"
            onClick={(e) => {
              e.preventDefault();
              // Load the first sample or notify
              onSelectSample(sampleFiles[0]);
            }}
          >
            <span>View More Samples</span>
            <ArrowRight size={15} />
          </a>
        </div>

        {/* 4 Cards Grid */}
        <div className="samples-grid">
          {sampleFiles.map((sample) => (
            <div key={sample.id} className="sample-card">
              <div className="sample-image-wrap">
                <img
                  src={sample.imageUrl}
                  alt={sample.title}
                  className="sample-thumb"
                  loading="lazy"
                />
                <span
                  className={`sample-status-tag ${sample.type === 'REAL' ? 'real' : 'fake'}`}
                >
                  {sample.type === 'REAL' ? 'Real' : 'Fake'}
                </span>
              </div>

              <div className="sample-content">
                <div className="sample-caption">{sample.title}</div>
                <button
                  type="button"
                  className="btn-try-this"
                  onClick={() => onSelectSample(sample)}
                >
                  Try This
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
