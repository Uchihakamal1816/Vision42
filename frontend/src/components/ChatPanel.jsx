import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Trash2 } from 'lucide-react';

function ChatPanel({ results, loading, onSendMessage, onClearChat }) {
  const [message, setMessage] = useState('');
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [results]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !loading) {
      onSendMessage(message);
      setMessage('');
    }
  };

  return (
    <div className="h-full bg-gray-50/30 dark:bg-black/15 backdrop-blur-lg rounded-xl border border-gray-200/30 dark:border-gray-700/20 flex flex-col shadow-lg shadow-black/5">
      {/* Header with Clear Button - More transparent */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200/30 dark:border-gray-700/20 bg-white/20 dark:bg-black/10 backdrop-blur-sm">
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-100 drop-shadow-sm">
          Chat Here!
        </h3>
        {results.length > 0 && (
          <button
            onClick={onClearChat}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-red-600 dark:text-red-300 hover:bg-red-50/50 dark:hover:bg-red-900/20 rounded-lg transition-colors backdrop-blur-sm"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear
          </button>
        )}
      </div>

      {/* Messages Area - SCROLLABLE with transparency */}
      <div 
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-5 space-y-5 scroll-smooth"
        style={{
          maxHeight: 'calc(100vh - 300px)',
          overflowY: 'auto'
        }}
      >
        {results.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center space-y-2">
              <p className="text-sm text-gray-600 dark:text-gray-300 font-medium drop-shadow-sm">No messages yet</p>
              <p className="text-xs text-gray-500 dark:text-gray-400 drop-shadow-sm">Upload an image and ask a question to get started</p>
            </div>
          </div>
        ) : (
          results.map((result, index) => (
            <div key={index} className="space-y-4">
              {/* User Message - Semi-transparent */}
              <div className="flex justify-end">
                <div className="bg-blue-600/90 dark:bg-blue-600/80 backdrop-blur-sm text-white rounded-2xl px-5 py-3 max-w-[85%] shadow-lg">
                  <p className="text-sm leading-relaxed font-medium">{result.query}</p>
                </div>
              </div>

              {/* Bot Response - More transparent */}
              {result.response ? (
                <div className="flex justify-start">
                  <div className="bg-white/50 dark:bg-black/25 backdrop-blur-md rounded-2xl px-5 py-3 max-w-[85%] border border-gray-200/40 dark:border-gray-600/30 shadow-lg">
                    {result.response.type === 'image' ? (
                      <div className="space-y-2">
                        <img 
                          src={result.response.imageUrl} 
                          alt="Detection result" 
                          className="rounded-xl max-w-full h-auto border border-gray-200/50 dark:border-gray-600/30 shadow-md"
                        />
                      </div>
                    ) : (
                      <p className="text-sm leading-relaxed text-gray-900 dark:text-gray-50 whitespace-pre-wrap drop-shadow-sm">
                        {result.response.text}
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="flex justify-start">
                  <div className="bg-white/50 dark:bg-black/25 backdrop-blur-md rounded-2xl px-5 py-3 border border-gray-200/40 dark:border-gray-600/30 shadow-lg">
                    <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                  </div>
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - More transparent */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200/30 dark:border-gray-700/20 bg-gray-50/20 dark:bg-black/10 backdrop-blur-sm">
        <div className="flex gap-3">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask a question about the image..."
            disabled={loading}
            className="flex-1 px-4 py-3 rounded-xl bg-white/60 dark:bg-black/30 backdrop-blur-md border border-gray-300/40 dark:border-gray-600/30 text-gray-900 dark:text-gray-50 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent disabled:opacity-50 transition-all text-sm shadow-sm"
          />
          <button
            type="submit"
            disabled={loading || !message.trim()}
            className="px-5 py-3 bg-blue-600/90 hover:bg-blue-700/90 backdrop-blur-sm text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shadow-lg font-medium"
          >
            {loading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

export default ChatPanel;
