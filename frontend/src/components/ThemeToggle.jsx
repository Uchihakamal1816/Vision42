import { useState, useEffect } from 'react';
import { Moon, Sun } from 'lucide-react';

export default function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    // Initialize from localStorage or default to 'dark'
    return localStorage.getItem('theme') || 'dark';
  });

  useEffect(() => {
    const root = window.document.documentElement;
    
    // Remove both classes first to avoid conflicts
    root.classList.remove('light', 'dark');
    
    // Add the current theme class
    root.classList.add(theme);
    
    // Persist to localStorage
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
  };

  return (
    <button
      onClick={toggleTheme}
      className="relative inline-flex h-10 w-20 items-center rounded-full bg-satellite-card border border-satellite-border transition-colors hover:bg-space-800"
    >
      <span className="sr-only">Toggle theme</span>
      <span
        className={`${
          theme === 'dark' ? 'translate-x-10' : 'translate-x-1'
        } inline-block h-8 w-8 transform rounded-full bg-space-500 transition-transform flex items-center justify-center shadow-glow`}
      >
        {theme === 'dark' ? (
          <Moon className="h-8 w-8 text-white" />
        ) : (
          <Sun className="h-8 w-8 text-white" />
        )}
      </span>
      <div className="absolute inset-0 flex items-center justify-between px-2 text-xs font-medium text-gray-400 pointer-events-none">
        <span className={theme === 'light' ? 'opacity-0' : ''}>Dark</span>
        <span className={theme === 'dark' ? 'opacity-0' : ''}>Light</span>
      </div>
    </button>
  );
}
