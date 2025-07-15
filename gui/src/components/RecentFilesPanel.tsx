import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  History, 
  Archive, 
  Download, 
  Clock, 
  FileText,
  X,
  CheckCircle,
  AlertCircle,
  Trash2
} from 'lucide-react'
import { RecentFile } from '../hooks/useRecentFiles'

interface RecentFilesPanelProps {
  recentFiles: RecentFile[]
  onRemoveFile: (id: string) => void
  onClearAll: () => void
  onFileSelect?: (filePath: string) => void
  className?: string
}

export const RecentFilesPanel: React.FC<RecentFilesPanelProps> = ({
  recentFiles,
  onRemoveFile,
  onClearAll,
  onFileSelect,
  className = ''
}) => {
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp)
    const now = new Date()
    const diff = now.getTime() - date.getTime()
    
    const minutes = Math.floor(diff / (1000 * 60))
    const hours = Math.floor(diff / (1000 * 60 * 60))
    const days = Math.floor(diff / (1000 * 60 * 60 * 24))
    
    if (minutes < 1) return 'たった今'
    if (minutes < 60) return `${minutes}分前`
    if (hours < 24) return `${hours}時間前`
    if (days < 7) return `${days}日前`
    return date.toLocaleDateString('ja-JP')
  }

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '-'
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  if (recentFiles.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`glass rounded-xl p-6 text-center ${className}`}
      >
        <History size={48} className="mx-auto mb-4 text-white/30" />
        <h4 className="text-lg font-semibold text-white/60 mb-2">履歴がありません</h4>
        <p className="text-white/50">ファイルを圧縮または展開すると、ここに履歴が表示されます</p>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass rounded-xl p-6 ${className}`}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <History size={18} className="text-blue-300" />
          <h4 className="font-semibold text-blue-100">最近のファイル</h4>
          <span className="text-sm text-white/60">({recentFiles.length})</span>
        </div>
        {recentFiles.length > 0 && (
          <motion.button
            onClick={onClearAll}
            className="flex items-center gap-2 text-sm text-red-300 hover:text-red-200 transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Trash2 size={14} />
            すべて削除
          </motion.button>
        )}
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        <AnimatePresence>
          {recentFiles.map((file, index) => (
            <motion.div
              key={file.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -100 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="relative group"
            >
              <div 
                className={`p-4 rounded-lg border transition-all cursor-pointer ${
                  file.success 
                    ? 'bg-white/5 border-white/10 hover:bg-white/10 hover:border-white/20' 
                    : 'bg-red-500/10 border-red-500/30 hover:bg-red-500/20'
                }`}
                onClick={() => onFileSelect?.(file.filePath)}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-start gap-3 flex-1 min-w-0">
                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                      file.operation === 'compress' 
                        ? 'bg-blue-500/20 text-blue-300' 
                        : 'bg-green-500/20 text-green-300'
                    }`}>
                      {file.operation === 'compress' ? (
                        <Archive size={16} />
                      ) : (
                        <Download size={16} />
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h5 className="font-medium text-white truncate">
                          {file.fileName}
                        </h5>
                        {file.success ? (
                          <CheckCircle size={14} className="text-green-400 flex-shrink-0" />
                        ) : (
                          <AlertCircle size={14} className="text-red-400 flex-shrink-0" />
                        )}
                      </div>
                      
                      <p className="text-sm text-white/70 truncate mb-2">
                        {file.filePath}
                      </p>
                      
                      <div className="flex items-center gap-4 text-xs text-white/60">
                        <div className="flex items-center gap-1">
                          <Clock size={12} />
                          {formatDate(file.timestamp)}
                        </div>
                        
                        {file.fileSize && (
                          <div className="flex items-center gap-1">
                            <FileText size={12} />
                            {formatFileSize(file.fileSize)}
                          </div>
                        )}
                        
                        {file.compressionRatio && (
                          <div className="flex items-center gap-1">
                            <Archive size={12} />
                            圧縮率: {file.compressionRatio.toFixed(1)}%
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <motion.button
                    onClick={(e) => {
                      e.stopPropagation()
                      onRemoveFile(file.id)
                    }}
                    className="opacity-0 group-hover:opacity-100 p-1 rounded-full hover:bg-white/10 text-white/50 hover:text-white/80 transition-all"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <X size={14} />
                  </motion.button>
                </div>
                
                {/* Success/Error indicator bar */}
                <div className={`absolute bottom-0 left-0 right-0 h-1 rounded-b-lg ${
                  file.success 
                    ? 'bg-gradient-to-r from-green-500 to-blue-500' 
                    : 'bg-gradient-to-r from-red-500 to-orange-500'
                }`} />
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}