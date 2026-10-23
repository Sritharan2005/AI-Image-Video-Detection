import React, { useState, useRef, useCallback } from 'react';

export default function ComparisonSlider({
  realImage = '/images/hero-face.jpg',
  fakeImage = '/images/hero-face-fake.jpg',
}) {
  const [splitPercent, setSplitPercent] = useState(50);
  const containerRef = useRef(null);
  const isDragging = useRef(false);

  const updatePosition = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    let percentage = (x / rect.width) * 100;
    if (percentage < 5) percentage = 5;
    if (percentage > 95) percentage = 95;
    setSplitPercent(percentage);
  }, []);

  const handleMouseDown = (e) => {
    isDragging.current = true;
    updatePosition(e.clientX);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  const handleMouseMove = useCallback((e) => {
    if (!isDragging.current) return;
    updatePosition(e.clientX);
  }, [updatePosition]);

  const handleMouseUp = useCallback(() => {
    isDragging.current = false;
    window.removeEventListener('mousemove', handleMouseMove);
    window.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseMove]);

  const handleTouchStart = (e) => {
    isDragging.current = true;
    if (e.touches[0]) updatePosition(e.touches[0].clientX);
  };

  const handleTouchMove = (e) => {
    if (!isDragging.current || !e.touches[0]) return;
    updatePosition(e.touches[0].clientX);
  };

  const handleTouchEnd = () => {
    isDragging.current = false;
  };

  return (
    <div
      ref={containerRef}
      className="comparison-container"
      style={{ '--split-percent': `${splitPercent}%` }}
      onMouseDown={handleMouseDown}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {/* Real Image (Base Left) */}
      <div className="comparison-image-wrapper">
        <img
          src={realImage}
          alt="Authentic human face"
          className="comparison-image"
          draggable={false}
        />
        <div className="comparison-badge real">Real</div>
      </div>

      {/* Fake Image (Clipped Right) */}
      <div className="comparison-image-wrapper comparison-image-fake">
        <img
          src={fakeImage}
          alt="AI manipulated face with artifacts"
          className="comparison-image"
          draggable={false}
        />
        <div className="comparison-badge fake">Fake</div>
      </div>

      {/* Center Draggable Slider Line & Handle */}
      <div className="comparison-slider-handle">
        <div className="comparison-slider-button" aria-label="Drag comparison slider">
          ❮ ❯
        </div>
      </div>
    </div>
  );
}
