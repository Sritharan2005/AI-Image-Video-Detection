import React from 'react';
import { UploadCloud, Settings, FileText, BarChart3, ArrowRight } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      num: '1',
      icon: <UploadCloud size={28} strokeWidth={2.2} />,
      title: 'Upload',
      desc: 'Upload an image or video file from your device.',
    },
    {
      num: '2',
      icon: <Settings size={28} strokeWidth={2.2} />,
      title: 'AI Analysis',
      desc: 'Our AI model analyzes the content for signs of manipulation.',
    },
    {
      num: '3',
      icon: <FileText size={28} strokeWidth={2.2} />,
      title: 'Get Results',
      desc: 'View the prediction with confidence score and detailed analysis.',
    },
    {
      num: '4',
      icon: <BarChart3 size={28} strokeWidth={2.2} />,
      title: 'Make Informed Decisions',
      desc: 'Use the results to verify content authenticity.',
    },
  ];

  return (
    <section id="how-it-works" className="how-it-works-section">
      <div className="container">
        <h2 className="how-title">How It Works</h2>

        <div className="how-steps-row">
          {steps.map((step, index) => (
            <React.Fragment key={step.num}>
              <div className="how-step-card">
                <div className="how-step-number">{step.num}</div>
                <div className="how-icon-box">{step.icon}</div>
                <h3 className="how-step-title">{step.title}</h3>
                <p className="how-step-desc">{step.desc}</p>
              </div>

              {index < steps.length - 1 && (
                <div className="how-arrow-indicator" aria-hidden="true">
                  <ArrowRight size={20} color="#0D5BFF" strokeWidth={2.5} />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>
    </section>
  );
}
