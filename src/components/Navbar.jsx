import React, { useState } from 'react';
import { ShieldCheck, Sun, Moon, Menu, X } from 'lucide-react';

export default function Navbar({ theme, onToggleTheme, activeSection, onNavigate }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navItems = [
    { label: 'Home', id: 'home' },
    { label: 'Detect', id: 'upload-section' },
    { label: 'How It Works', id: 'how-it-works' },
    { label: 'About', id: 'about-footer' },
  ];

  const handleNavClick = (id) => {
    onNavigate(id);
    setMobileMenuOpen(false);
  };

  return (
    <header className="navbar-header">
      <div className="container navbar-container">
        {/* Brand Logo */}
        <a href="#home" onClick={(e) => { e.preventDefault(); handleNavClick('home'); }} className="brand-logo">
          <div className="brand-logo-icon">
            <ShieldCheck size={28} strokeWidth={2.4} />
          </div>
          <span className="brand-name">
            Deep<span>Guard</span>
          </span>
        </a>

        {/* Center Nav Links */}
        <nav>
          <ul className="nav-links">
            {navItems.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  onClick={() => handleNavClick(item.id)}
                  className={`nav-link ${activeSection === item.id ? 'active' : ''}`}
                >
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        {/* Right Actions */}
        <div className="nav-actions">
          <button
            type="button"
            className="theme-toggle-btn"
            onClick={onToggleTheme}
            aria-label="Toggle dark/light theme"
            title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
          >
            {theme === 'dark' ? <Sun size={19} /> : <Sun size={19} />}
          </button>

          <button
            type="button"
            className="btn-primary"
            onClick={() => handleNavClick('upload-section')}
          >
            Try Now
          </button>

          <button
            type="button"
            className="mobile-menu-btn"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle Menu"
          >
            {mobileMenuOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="mobile-nav-drawer">
          {navItems.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => handleNavClick(item.id)}
              className={`nav-link ${activeSection === item.id ? 'active' : ''}`}
              style={{ textAlign: 'left', padding: '10px 0' }}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}
    </header>
  );
}
