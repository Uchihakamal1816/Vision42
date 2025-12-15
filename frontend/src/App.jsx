import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ThemeToggle from './components/ThemeToggle';
import ImageDropZone from './components/ImageDropZone';
import ChatPanel from './components/ChatPanel';
import Galaxy from './components/Galaxy';
import DotGrid from './components/DotGrid';
import './index.css'
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:15200';

function App() {
  const [sessions, setSessions] = useState([]);
  const [activeSession, setActiveSession] = useState(null);
  const [image, setImage] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });

  // Watch for theme changes via MutationObserver (handles external theme toggles)
  useEffect(() => {
    const observer = new MutationObserver(() => {
      const isDark = document.documentElement.classList.contains('dark');
      setTheme(isDark ? 'dark' : 'light');
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    return () => observer.disconnect();
  }, []);

  const handleNewSession = () => {
    const newSession = {
      id: Date.now(),
      name: `Session ${sessions.length + 1}`,
      timestamp: new Date().toLocaleTimeString(),
    };

    setSessions([newSession, ...sessions]);
    setActiveSession(newSession.id);

    // Reset state for fresh session
    setImage(null);
    setResults([]);
    setLoading(false);
  };

  const handleClearChat = () => {
    setResults([]);
  };

  const handleImageUpload = async (file) => {
    const reader = new FileReader();
    reader.onload = () => {
      setImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleSendMessage = async (message) => {
    if (!image) {
      alert('Please upload an image first');
      return;
    }

    setLoading(true);
    setResults([...results, { query: message, response: null }]);

    try {
      // Convert base64 → blob for FormData
      const formData = new FormData();
      const blob = await fetch(image).then(r => r.blob());
      formData.append('file', blob, 'satellite.jpg');
      formData.append('query', message);
      formData.append('query_type', 'auto');

      const response = await axios.post(`${API_URL}/geoNLI/eval`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        responseType: 'blob', // Could be image or JSON
        timeout: 180000
      });

      console.log('Response headers:', response.headers);
      console.log('Response type:', response.headers['content-type']);

      let answer;

      // Handle grounding query (returns annotated image)
      if (response.headers['content-type']?.includes('image')) {
        console.log('Received image response');

        const imageUrl = URL.createObjectURL(response.data);

        answer = {
          type: 'image',
          imageUrl: imageUrl
        };
      } 
      // Handle text-based queries (caption, numeric, binary, semantic)
      else if (response.headers['content-type']?.includes('json')) {
        console.log('Received JSON response');

        // Blob → JSON conversion
        const text = await new Response(response.data).text();

        let jsonData = null;
        try {
          jsonData = JSON.parse(text);
        } catch (e) {
          console.error("JSON parse failed:", e);
        }

        // Graceful handling of empty/invalid responses
        if (!jsonData || typeof jsonData.response === "undefined" || jsonData.response === null) {
          answer = {
            type: 'text',
            text: 'No valid response from server.'
          };
        } else {
          answer = {
            type: 'text',
            text: jsonData.response
          };
        }
      }
      else {
        console.error('Unknown response type:', response.headers['content-type']);
        answer = {
          type: 'text',
          text: 'Error: Unknown response format'
        };
      }

      // Update last result with response
      setResults(prev => {
        const updated = [...prev];
        updated[updated.length - 1].response = answer;
        return updated;
      });

    } catch (error) {
      console.error('Error:', error);

      let errorMessage = 'Server error';

      if (error.response) {
        try {
          const text = await error.response.data.text();
          const jsonError = JSON.parse(text);
          errorMessage = jsonError.detail || 'Server error';
        } catch (e) {
          errorMessage = error.message;
        }
      } else {
        errorMessage = error.message;
      }

      setResults(prev => {
        const updated = [...prev];
        updated[updated.length - 1].response = {
          type: 'text',
          text: 'Error: ' + errorMessage
        };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen flex bg-white dark:bg-gray-950 relative overflow-hidden">
      {/* Dark mode: Galaxy animation background */}
      {theme === 'dark' && (
        <div className="absolute inset-0 opacity-70 pointer-events-none">
          <Galaxy
            mouseRepulsion={false}
            mouseInteraction={false}
            density={0.65}            
            glowIntensity={0.35}     
            twinkleIntensity={0.55}  
            speed={0.19}          
            rotationSpeed={0.08}    
            hueShift={200}
            transparent={true}
            fps={30} // Performance optimization              
          />
        </div>
      )}

      {/* Light mode: Interactive dot grid */}
      {theme === 'light' && (
        <div className="absolute inset-0 opacity-40 pointer-events-none">
          <DotGrid
            dotSize={8}
            gap={20}
            baseColor="#94a3b8"
            activeColor="#3b82f6"
            proximity={100}
            speedTrigger={80}
            shockRadius={200}
            shockStrength={4}
            resistance={600}
            returnDuration={1.2}
          />
        </div>
      )}

      {/* Main UI layer */}
      <div className="relative z-10 flex w-full h-full">
        <Sidebar
          sessions={sessions}
          activeSession={activeSession}
          onNewSession={handleNewSession}
          onSelectSession={setActiveSession}
        />

        {/* Main content area */}
        <div className="flex-1 flex flex-col p-6 gap-6">
          {/* Header with branding + theme toggle */}
          <div className="flex items-center justify-between bg-white/40 dark:bg-black/20 backdrop-blur-md rounded-xl p-5 border border-gray-200/30 dark:border-gray-700/20 shadow-lg shadow-black/5">
            <div className="flex items-center gap-3">
              <div className="h-11 w-11 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-lg shadow-blue-500/30">
                <span className="text-white font-bold text-xl">V</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900 dark:text-white drop-shadow-sm">Vision42 GeoNLi</h1>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-gray-600 dark:text-gray-200 drop-shadow-sm">Theme:</span>
              <ThemeToggle />
            </div>
          </div>

          {/* Two-column layout: Image viewer + Chat */}
          <div className="flex-1 grid grid-cols-2 gap-6 min-h-0">
            {/* Left: Image upload/preview */}
            <div className="space-y-3 flex flex-col">
              <div className="flex items-center justify-between">
                {image && (
                  <button
                    onClick={() => setImage(null)}
                    className="text-xs font-medium text-blue-600 dark:text-blue-300 hover:text-blue-700 dark:hover:text-blue-200 transition-colors drop-shadow-sm"
                  >
                    Clear
                  </button>
                )}
              </div>
              <div className="flex-1 bg-gray-50/30 dark:bg-black/15 backdrop-blur-lg rounded-xl border border-gray-200/30 dark:border-gray-700/20 p-5 shadow-lg shadow-black/5">
                <ImageDropZone image={image} onImageUpload={handleImageUpload} />
              </div>
            </div>

            {/* Right: Chat interface */}
            <div className="space-y-3 flex flex-col min-h-0">
              <div className="flex-1 min-h-0">
                <ChatPanel
                  results={results}
                  loading={loading}
                  onSendMessage={handleSendMessage}
                  onClearChat={handleClearChat}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;