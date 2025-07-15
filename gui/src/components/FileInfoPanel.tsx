import React from 'react'
import { motion } from 'framer-motion'
import { 
  Archive, 
  Shield, 
  Clock, 
  HardDrive, 
  Zap,
  Lock,
  Unlock,
  FileText,
  Info
} from 'lucide-react'
import { FileInfo } from '../hooks/useNXZip'

interface FileInfoPanelProps {
  fileInfo: FileInfo | null
  isLoading?: boolean
  className?: string
}

export const FileInfoPanel: React.FC<FileInfoPanelProps> = ({ 
  fileInfo, 
  isLoading = false, 
  className = '' 
}) => {
  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`glass rounded-xl p-6 ${className}`}
      >
        <div className="flex items-center gap-2 mb-4">
          <Info size={18} className="text-blue-300" />
          <h4 className="font-semibold text-blue-100">ファイル情報を読み込み中...</h4>
        </div>
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="animate-pulse">
              <div className="h-4 bg-white/10 rounded mb-2"></div>
              <div className="h-3 bg-white/5 rounded w-2/3"></div>
            </div>
          ))}
        </div>
      </motion.div>
    )
  }

  if (!fileInfo) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={`glass rounded-xl p-6 text-center ${className}`}
      >
        <FileText size={48} className="mx-auto mb-4 text-white/30" />
        <p className="text-white/60">ファイルを選択すると詳細情報が表示されます</p>
      </motion.div>
    )
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString('ja-JP')
    } catch {
      return dateString
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass rounded-xl p-6 ${className}`}
    >
      <div className="flex items-center gap-2 mb-6">
        <Archive size={18} className="text-blue-300" />
        <h4 className="font-semibold text-blue-100">ファイル情報</h4>
      </div>

      <div className="space-y-4">
        {/* File Name */}
        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-2">
            <FileText size={16} className="text-blue-300" />
            <span className="text-white/80 font-medium">ファイル名</span>
          </div>
          <span className="text-white font-medium truncate max-w-48" title={fileInfo.filename}>
            {fileInfo.filename}
          </span>
        </div>

        {/* File Sizes */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-2">
              <HardDrive size={16} className="text-green-300" />
              <span className="text-white/80 font-medium">元サイズ</span>
            </div>
            <span className="text-green-300 font-medium">
              {formatFileSize(fileInfo.originalSize)}
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-2">
              <Archive size={16} className="text-purple-300" />
              <span className="text-white/80 font-medium">圧縮後</span>
            </div>
            <span className="text-purple-300 font-medium">
              {formatFileSize(fileInfo.compressedSize)}
            </span>
          </div>
        </div>

        {/* Compression Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-2">
              <Zap size={16} className="text-yellow-300" />
              <span className="text-white/80 font-medium">圧縮率</span>
            </div>
            <span className="text-yellow-300 font-medium">
              {fileInfo.compressionRatio.toFixed(1)}%
            </span>
          </div>
          
          <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
            <div className="flex items-center gap-2">
              <Zap size={16} className="text-orange-300" />
              <span className="text-white/80 font-medium">アルゴリズム</span>
            </div>
            <span className="text-orange-300 font-medium">
              {fileInfo.algorithm}
            </span>
          </div>
        </div>

        {/* Encryption Status */}
        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-2">
            {fileInfo.isEncrypted ? (
              <Lock size={16} className="text-red-300" />
            ) : (
              <Unlock size={16} className="text-gray-300" />
            )}
            <span className="text-white/80 font-medium">暗号化</span>
          </div>
          <div className="flex items-center gap-2">
            {fileInfo.isEncrypted ? (
              <>
                <span className="text-red-300 font-medium">有効</span>
                {fileInfo.encryptionAlgorithm && (
                  <span className="text-xs text-red-200 bg-red-500/20 px-2 py-1 rounded">
                    {fileInfo.encryptionAlgorithm}
                  </span>
                )}
              </>
            ) : (
              <span className="text-gray-400 font-medium">無効</span>
            )}
          </div>
        </div>

        {/* Creation Time */}
        <div className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
          <div className="flex items-center gap-2">
            <Clock size={16} className="text-blue-300" />
            <span className="text-white/80 font-medium">作成日時</span>
          </div>
          <span className="text-blue-300 font-medium">
            {formatDate(fileInfo.createdAt)}
          </span>
        </div>

        {/* Compression Efficiency Indicator */}
        {fileInfo.compressionRatio > 0 && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-white/80">圧縮効率</span>
              <span className="text-sm text-white/60">
                {100 - fileInfo.compressionRatio > 0 
                  ? `${(100 - fileInfo.compressionRatio).toFixed(1)}% 削減`
                  : '圧縮なし'
                }
              </span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-2">
              <motion.div
                className="h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                initial={{ width: 0 }}
                animate={{ width: `${Math.max(0, 100 - fileInfo.compressionRatio)}%` }}
                transition={{ duration: 1, delay: 0.5 }}
              />
            </div>
          </div>
        )}
      </div>
    </motion.div>
  )
}