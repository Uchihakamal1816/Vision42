 import { Plus, History, MessageSquare } from 'lucide-react';
import { motion } from 'framer-motion';

export default function Sidebar({ sessions, activeSession, onNewSession, onSelectSession }) {
  return (
    <motion.div
      initial={{ x: -300 }}
      animate={{ x: 0 }}
      className="w-64 h-screen bg-white/40 dark:bg-black/20 backdrop-blur-lg border-r border-gray-200/30 dark:border-gray-700/20 flex flex-col transition-colors shadow-xl shadow-black/5"
    >
      {/* Header */}
      <div className="p-5 border-b border-gray-200/30 dark:border-gray-700/20">
        <div className="flex items-center gap-3 mb-5">
          <div className="h-9 w-9 rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-md shadow-blue-500/30">
            V
          </div>
          <h1 className="text-lg font-bold bg-gradient-to-r from-blue-600 to-blue-700 dark:from-blue-400 dark:to-blue-500 bg-clip-text text-transparent drop-shadow-sm">
            Vision42
          </h1>
        </div>

        {/* New Session Button */}
        <button
          onClick={onNewSession}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600/90 hover:bg-blue-700/90 dark:bg-blue-600/80 dark:hover:bg-blue-700/80 backdrop-blur-sm text-white rounded-xl transition-all shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] font-medium"
        >
          <Plus className="h-4 w-4" />
          <span>New Session</span>
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="flex items-center gap-2 mb-4 px-1">
          <History className="h-4 w-4 text-gray-600 dark:text-gray-300" />
          <h3 className="text-xs font-semibold text-gray-600 dark:text-gray-200 uppercase tracking-wider drop-shadow-sm">Chat History</h3>
        </div>

        {sessions.length === 0 ? (
          <div className="bg-gray-100/40 dark:bg-black/20 backdrop-blur-sm rounded-xl p-5 border border-gray-200/30 dark:border-gray-700/20">
            <p className="text-sm font-medium text-gray-600 dark:text-gray-200 drop-shadow-sm">Empty session</p>
            <p className="text-xs text-gray-500 dark:text-gray-300 mt-1.5 drop-shadow-sm">No messages yet</p>
          </div>
        ) : (
          <div className="space-y-2">
            {sessions.map((session) => (
              <motion.button
                key={session.id}
                whileHover={{ scale: 1.02, x: 4 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onSelectSession(session.id)}
                className={`w-full text-left p-3.5 rounded-xl transition-all ${
                  activeSession === session.id
                    ? 'bg-blue-50/60 dark:bg-blue-900/20 backdrop-blur-sm border border-blue-200/50 dark:border-blue-700/30 shadow-md'
                    : 'bg-gray-50/40 dark:bg-black/15 backdrop-blur-sm hover:bg-gray-100/50 dark:hover:bg-black/25 border border-gray-200/30 dark:border-gray-700/20'
                }`}
              >
                <div className="flex items-start gap-3">
                  <MessageSquare className={`h-4 w-4 mt-0.5 flex-shrink-0 ${
                    activeSession === session.id 
                      ? 'text-blue-600 dark:text-blue-400' 
                      : 'text-gray-600 dark:text-gray-300'
                  }`} />
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-semibold truncate drop-shadow-sm ${
                      activeSession === session.id
                        ? 'text-gray-900 dark:text-gray-50'
                        : 'text-gray-700 dark:text-gray-100'
                    }`}>
                      {session.name}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-300 mt-0.5 drop-shadow-sm">
                      {session.timestamp}
                    </p>
                  </div>
                </div>
              </motion.button>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
