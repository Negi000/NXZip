import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Archive, 
  Shield, 
  Zap, 
  Upload, 
  Download, 
  Settings, 
  Info,
  Unlock,
  FileText,
  FolderOpen,
  Star,
  Cpu,
  HardDrive,
  AlertCircle,
  CheckCircle
} from 'lucide-react'
import { useNXZip } from './hooks/useNXZip'

// Particle component for background animation
const Particle: React.FC<{ delay: number }> = ({ delay }) => {
  const [position, setPosition] = useState({ x: Math.random() * window.innerWidth, y: window.innerHeight })
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setPosition({ x: Math.random() * window.innerWidth, y: -100 })
    }, delay * 1000)
    
    return () => clearTimeout(timer)
  }, [delay])

  return (
    <motion.div
      className="particle"
      style={{
        left: position.x,
        top: position.y,
        width: Math.random() * 4 + 2,
        height: Math.random() * 4 + 2,
      }}
      animate={{
        y: -window.innerHeight - 100,
        x: position.x + (Math.random() - 0.5) * 100,
      }}
      transition={{
        duration: 6,
        repeat: Infinity,
        delay: delay,
      }}
    />
  )
}

// Main Tab Component
interface TabProps {
  id: string
  title: string
  icon: React.ReactNode
  isActive: boolean
  onClick: () => void
}

// Enhanced Tab Component with Better Visual Feedback
const Tab: React.FC<TabProps> = ({ title, icon, isActive, onClick }) => (
  <motion.button
    className={`relative flex items-center gap-3 px-8 py-4 rounded-xl transition-all duration-300 font-semibold ${
      isActive 
        ? 'glass neon-glow bg-emerald-500/20 border-2 border-emerald-400/50 text-emerald-100 shadow-lg shadow-emerald-500/20' 
        : 'glass-dark hover:bg-white/10 text-white/80 hover:text-white border-2 border-transparent hover:border-emerald-400/20'
    }`}
    onClick={onClick}
    whileHover={{ scale: 1.05, y: -2 }}
    whileTap={{ scale: 0.95 }}
    layout
  >
    <motion.span 
      className="text-xl"
      animate={{ 
        color: isActive ? '#6ee7b7' : '#ffffff',
        scale: isActive ? 1.1 : 1 
      }}
    >
      {icon}
    </motion.span>
    <span className="font-medium">{title}</span>
    {isActive && (
      <motion.div
        className="absolute bottom-0 left-1/2 w-8 h-1 bg-emerald-400 rounded-full"
        layoutId="activeIndicator"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        style={{ transform: 'translateX(-50%)' }}
      />
    )}
  </motion.button>
)

// Compression Tab Component
const CompressionTab: React.FC = () => {
  const { compressFile, selectMultipleFiles, isLoading, error } = useNXZip()
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [progress, setProgress] = useState(0)
  const [isCompressing, setIsCompressing] = useState(false)
  const [compressionOptions, setCompressionOptions] = useState({
    algorithm: 'auto',
    level: 6,
    encryption: '',
    password: '',
    kdf: 'argon2',
    outputPath: ''
  })
  const [result, setResult] = useState<any>(null)

  const handleFileSelect = async () => {
    const files = await selectMultipleFiles()
    if (files) {
      setSelectedFiles(files)
    }
  }

  const handleCompress = async () => {
    if (selectedFiles.length === 0) return

    setIsCompressing(true)
    setProgress(0)
    setResult(null)

    try {
      // プログレスのシミュレーション
      const progressInterval = setInterval(() => {
        setProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return 90
          }
          return prev + Math.random() * 15
        })
      }, 200)

      const inputPath = selectedFiles[0] // 最初のファイルを使用
      const outputPath = compressionOptions.outputPath || `${inputPath}.nxz`

      const options = {
        algorithm: compressionOptions.algorithm,
        level: compressionOptions.level,
        encryption: compressionOptions.encryption || undefined,
        password: compressionOptions.password || undefined,
        kdf: compressionOptions.kdf,
      }

      const compressResult = await compressFile(inputPath, outputPath, options)
      
      clearInterval(progressInterval)
      setProgress(100)
      setResult(compressResult)
    } catch (err) {
      console.error('Compression error:', err)
    } finally {
      setIsCompressing(false)
      setTimeout(() => setProgress(0), 2000)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-4 border-red-500/50"
        >
          <div className="flex items-center gap-2 text-red-300">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        </motion.div>
      )}

      {/* Success Display */}
      {result && result.success && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-4 border-green-500/50"
        >
          <div className="flex items-center gap-2 text-green-300 mb-2">
            <CheckCircle size={18} />
            <span>{result.message}</span>
          </div>
          {result.compressionRatio && (
            <div className="text-sm text-white/70">
              圧縮率: {result.compressionRatio.toFixed(1)}% | 
              元サイズ: {(result.originalSize / 1024).toFixed(1)} KB | 
              圧縮後: {(result.compressedSize / 1024).toFixed(1)} KB
            </div>
          )}
        </motion.div>
      )}

      {/* Enhanced File Selection Area */}
      <motion.div
        className="relative p-12 border-3 border-dashed rounded-2xl transition-all duration-300 border-emerald-400/50 glass cursor-pointer hover:border-emerald-400 hover:bg-emerald-500/10 hover:shadow-lg hover:shadow-emerald-500/20"
        onClick={handleFileSelect}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <div className="text-center">
          <motion.div
            className="floating mx-auto mb-4 w-24 h-24 flex items-center justify-center glass rounded-full bg-emerald-500/20 border-2 border-emerald-400/30"
            whileHover={{ scale: 1.1, rotate: 5 }}
            animate={{ 
              boxShadow: ["0 0 20px rgba(16, 185, 129, 0.3)", "0 0 30px rgba(16, 185, 129, 0.5)", "0 0 20px rgba(16, 185, 129, 0.3)"] 
            }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            <Upload size={36} className="text-emerald-300" />
          </motion.div>
          <h3 className="text-2xl font-bold mb-2 text-emerald-200">ファイルをここにドロップ</h3>
          <p className="text-emerald-200/80 mb-4 text-lg">または、クリックしてファイルを選択</p>
          <motion.div 
            className="btn-primary inline-flex items-center bg-emerald-600 hover:bg-emerald-500 border-emerald-400"
            whileHover={{ scale: 1.05 }}
          >
            <FolderOpen size={20} className="mr-2" />
            ファイル選択
          </motion.div>
        </div>
        {/* Visual Enhancement Lines */}
        <div className="absolute inset-4 border border-emerald-400/20 rounded-xl pointer-events-none"></div>
        <div className="absolute inset-8 border border-emerald-400/10 rounded-lg pointer-events-none"></div>
      </motion.div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="glass rounded-xl p-4"
        >
          <h4 className="font-semibold mb-3 flex items-center">
            <FileText size={18} className="mr-2" />
            選択されたファイル ({selectedFiles.length})
          </h4>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between bg-white/10 rounded-lg p-2">
                <span className="truncate">{file.split('\\').pop() || file.split('/').pop()}</span>
                <span className="text-sm text-white/70">{file}</span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Compression Options */}
      <div className="glass rounded-xl p-6">
        <h4 className="font-semibold mb-4 flex items-center">
          <Settings size={18} className="mr-2" />
          圧縮設定
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">圧縮アルゴリズム</label>
            <select 
              className="input-field"
              value={compressionOptions.algorithm}
              onChange={(e) => setCompressionOptions(prev => ({ ...prev, algorithm: e.target.value }))}
            >
              <option value="auto">Auto (自動選択)</option>
              <option value="zstd">Zstd (高速)</option>
              <option value="lzma2">LZMA2 (高圧縮)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">圧縮レベル</label>
            <select 
              className="input-field"
              value={compressionOptions.level}
              onChange={(e) => setCompressionOptions(prev => ({ ...prev, level: parseInt(e.target.value) }))}
            >
              <option value={1}>1 (最高速)</option>
              <option value={6}>6 (バランス)</option>
              <option value={9}>9 (最高圧縮)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">暗号化</label>
            <select 
              className="input-field"
              value={compressionOptions.encryption}
              onChange={(e) => setCompressionOptions(prev => ({ ...prev, encryption: e.target.value }))}
            >
              <option value="">なし</option>
              <option value="aes-gcm">AES-256-GCM</option>
              <option value="xchacha20">XChaCha20-Poly1305</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">出力ファイル名</label>
            <input 
              type="text" 
              placeholder="archive.nxz" 
              className="input-field"
              value={compressionOptions.outputPath}
              onChange={(e) => setCompressionOptions(prev => ({ ...prev, outputPath: e.target.value }))}
            />
          </div>
          {compressionOptions.encryption && (
            <>
              <div>
                <label className="block text-sm font-medium mb-2">パスワード</label>
                <input 
                  type="password" 
                  placeholder="••••••••" 
                  className="input-field"
                  value={compressionOptions.password}
                  onChange={(e) => setCompressionOptions(prev => ({ ...prev, password: e.target.value }))}
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">鍵導出方式</label>
                <select 
                  className="input-field"
                  value={compressionOptions.kdf}
                  onChange={(e) => setCompressionOptions(prev => ({ ...prev, kdf: e.target.value }))}
                >
                  <option value="argon2">Argon2id (推奨)</option>
                  <option value="pbkdf2">PBKDF2-SHA256</option>
                </select>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      {isCompressing && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-4"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="font-medium">圧縮中...</span>
            <span className="text-sm">{Math.round(progress)}%</span>
          </div>
          <div className="progress-bar">
            <motion.div
              className="progress-fill"
              style={{ width: `${progress}%` }}
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
            />
          </div>
        </motion.div>
      )}

      {/* Action Buttons */}
      <div className="flex gap-4">
        <motion.button
          className="btn-primary flex-1 flex items-center justify-center gap-2"
          onClick={handleCompress}
          disabled={selectedFiles.length === 0 || isCompressing || isLoading}
          whileHover={{ scale: selectedFiles.length > 0 && !isCompressing ? 1.02 : 1 }}
          whileTap={{ scale: selectedFiles.length > 0 && !isCompressing ? 0.98 : 1 }}
        >
          <Archive size={18} />
          {isCompressing ? '圧縮中...' : '圧縮開始'}
        </motion.button>
        <motion.button
          className="btn-secondary flex items-center gap-2"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <Shield size={18} />
          セキュア圧縮
        </motion.button>
      </div>
    </motion.div>
  )
}

// Extraction Tab Component
const ExtractionTab: React.FC = () => {
  const { extractFile, selectFile, isLoading, error } = useNXZip()
  const [selectedFile, setSelectedFile] = useState<string>('')
  const [outputPath, setOutputPath] = useState<string>('')
  const [password, setPassword] = useState<string>('')
  const [result, setResult] = useState<any>(null)

  const handleFileSelect = async () => {
    const file = await selectFile([
      { name: 'NXZ Archives', extensions: ['nxz'] },
      { name: 'NXZ Secure Archives', extensions: ['nxz.sec'] },
      { name: 'All Files', extensions: ['*'] }
    ])
    if (file) {
      setSelectedFile(file)
      // 出力パスを自動生成
      const baseName = file.replace(/\.(nxz\.sec|nxz)$/, '')
      setOutputPath(`${baseName}_extracted`)
    }
  }

  const handleExtract = async () => {
    if (!selectedFile) return

    setResult(null)
    const extractResult = await extractFile(selectedFile, outputPath, password || undefined)
    setResult(extractResult)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Error Display */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-4 border-red-500/50"
        >
          <div className="flex items-center gap-2 text-red-300">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        </motion.div>
      )}

      {/* Success Display */}
      {result && result.success && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass rounded-xl p-4 border-green-500/50"
        >
          <div className="flex items-center gap-2 text-green-300 mb-2">
            <CheckCircle size={18} />
            <span>{result.message}</span>
          </div>
          {result.extractedSize && (
            <div className="text-sm text-white/70">
              展開サイズ: {(result.extractedSize / 1024).toFixed(1)} KB
            </div>
          )}
        </motion.div>
      )}

      {/* Enhanced File Selection for Extraction */}
      <motion.div 
        className="glass rounded-xl p-12 text-center cursor-pointer hover:bg-emerald-500/10 transition-all border-2 border-emerald-400/30 hover:border-emerald-400 hover:shadow-lg hover:shadow-emerald-500/20"
        onClick={handleFileSelect}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <motion.div
          className="floating mx-auto mb-4 w-24 h-24 flex items-center justify-center glass rounded-full bg-emerald-500/20 border-2 border-emerald-400/30"
          whileHover={{ scale: 1.1, rotate: -5 }}
          animate={{ 
            boxShadow: ["0 0 20px rgba(16, 185, 129, 0.3)", "0 0 30px rgba(16, 185, 129, 0.5)", "0 0 20px rgba(16, 185, 129, 0.3)"] 
          }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <Download size={36} className="text-emerald-300" />
        </motion.div>
        <h3 className="text-2xl font-bold mb-2 text-emerald-200">アーカイブを選択</h3>
        <p className="text-emerald-200/80 mb-4 text-lg">.nxz または .nxz.sec ファイルをドロップまたは選択</p>
        {selectedFile ? (
          <motion.div 
            className="glass rounded-lg p-4 mb-4 bg-emerald-500/20 border border-emerald-400/30"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <span className="text-sm text-emerald-200 font-semibold">選択済み: </span>
            <span className="text-sm text-emerald-100">{selectedFile.split('\\').pop() || selectedFile.split('/').pop()}</span>
          </motion.div>
        ) : null}
        <motion.div 
          className="btn-primary inline-flex items-center bg-emerald-600 hover:bg-emerald-500 border-emerald-400"
          whileHover={{ scale: 1.05 }}
        >
          <FolderOpen size={20} className="mr-2" />
          {selectedFile ? 'ファイル変更' : 'アーカイブ選択'}
        </motion.div>
      </motion.div>

      {/* Extraction Settings */}
      <div className="glass rounded-xl p-6">
        <h4 className="font-semibold mb-4 flex items-center">
          <Unlock size={18} className="mr-2" />
          展開設定
        </h4>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">出力ファイル/ディレクトリ</label>
            <input 
              type="text" 
              placeholder="./extracted_file" 
              className="input-field"
              value={outputPath}
              onChange={(e) => setOutputPath(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">パスワード (暗号化されている場合)</label>
            <input 
              type="password" 
              placeholder="••••••••" 
              className="input-field"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
        </div>
      </div>

      {/* Extract Button */}
      <motion.button 
        className="btn-primary w-full flex items-center justify-center gap-2"
        onClick={handleExtract}
        disabled={!selectedFile || !outputPath || isLoading}
        whileHover={{ scale: selectedFile && outputPath && !isLoading ? 1.02 : 1 }}
        whileTap={{ scale: selectedFile && outputPath && !isLoading ? 0.98 : 1 }}
      >
        <Download size={18} />
        {isLoading ? '展開中...' : '展開開始'}
      </motion.button>
    </motion.div>
  )
}

// Settings Tab Component
const SettingsTab: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="glass rounded-xl p-6">
        <h4 className="font-semibold mb-4 flex items-center">
          <Cpu size={18} className="mr-2" />
          パフォーマンス設定
        </h4>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">スレッド数</label>
            <select className="input-field">
              <option>自動 (推奨)</option>
              <option>1 スレッド</option>
              <option>2 スレッド</option>
              <option>4 スレッド</option>
              <option>8 スレッド</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">メモリ使用量</label>
            <select className="input-field">
              <option>中程度 (推奨)</option>
              <option>低</option>
              <option>高</option>
            </select>
          </div>
        </div>
      </div>

      <div className="glass rounded-xl p-6">
        <h4 className="font-semibold mb-4 flex items-center">
          <Shield size={18} className="mr-2" />
          セキュリティ設定
        </h4>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">デフォルト暗号化</label>
            <select className="input-field">
              <option>AES-256-GCM</option>
              <option>XChaCha20-Poly1305</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">鍵導出方式</label>
            <select className="input-field">
              <option>Argon2id (推奨)</option>
              <option>PBKDF2-SHA256</option>
            </select>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

// Enhanced About Tab Component with Better Contrast
const AboutTab: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="glass rounded-xl p-8 text-center border border-emerald-400/30">
        <motion.div
          className="floating mx-auto mb-6 w-28 h-28 flex items-center justify-center glass rounded-full neon-glow bg-emerald-500/20 border-2 border-emerald-400/50"
          whileHover={{ scale: 1.1, rotate: 360 }}
          transition={{ duration: 0.8 }}
        >
          <Star size={48} className="gradient-text" />
        </motion.div>
        <h2 className="text-4xl font-bold mb-3 gradient-text">NXZip</h2>
        <p className="text-2xl text-emerald-200 mb-4 font-semibold">次世代統合アーカイブシステム</p>
        <p className="text-emerald-300 font-medium text-lg">Version 1.0.0</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.div 
          className="card text-center border border-emerald-400/30 hover:border-emerald-400/60 transition-all"
          whileHover={{ scale: 1.05, y: -5 }}
        >
          <Archive size={40} className="mx-auto mb-4 text-emerald-300" />
          <h4 className="font-bold mb-3 text-emerald-100 text-lg">SPE暗号化</h4>
          <p className="text-emerald-200 font-medium">構造保持型暗号化技術</p>
        </motion.div>
        <motion.div 
          className="card text-center border border-yellow-400/30 hover:border-yellow-400/60 transition-all"
          whileHover={{ scale: 1.05, y: -5 }}
        >
          <Zap size={40} className="mx-auto mb-4 text-yellow-300" />
          <h4 className="font-bold mb-3 text-yellow-100 text-lg">高速圧縮</h4>
          <p className="text-yellow-200 font-medium">LZMA2/Zstd最適化</p>
        </motion.div>
        <motion.div 
          className="card text-center border border-emerald-400/30 hover:border-emerald-400/60 transition-all"
          whileHover={{ scale: 1.05, y: -5 }}
        >
          <Shield size={40} className="mx-auto mb-4 text-emerald-300" />
          <h4 className="font-bold mb-3 text-emerald-100 text-lg">多層セキュリティ</h4>
          <p className="text-emerald-200 font-medium">AES-GCM/XChaCha20</p>
        </motion.div>
      </div>

      <div className="glass rounded-xl p-6 border border-emerald-400/30">
        <h4 className="font-bold mb-6 flex items-center text-emerald-100 text-xl">
          <HardDrive size={24} className="mr-3 text-emerald-300" />
          システム情報
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-base">
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
              <span className="text-emerald-200 font-semibold">Rust バージョン:</span>
              <span className="text-emerald-100 font-medium">1.70.0+</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
              <span className="text-emerald-200 font-semibold">GUI フレームワーク:</span>
              <span className="text-emerald-100 font-medium">Tauri + React</span>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
              <span className="text-emerald-200 font-semibold">圧縮エンジン:</span>
              <span className="text-emerald-100 font-medium">Custom NXZ</span>
            </div>
            <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
              <span className="text-emerald-200 font-semibold">暗号化:</span>
              <span className="text-emerald-100 font-medium">AES-256, XChaCha20</span>
            </div>
          </div>
        </div>
      </div>

      {/* Additional Info Section */}
      <div className="glass rounded-xl p-6 border border-emerald-400/30">
        <h4 className="font-bold mb-4 flex items-center text-emerald-100 text-xl">
          <Info size={24} className="mr-3 text-emerald-300" />
          開発情報
        </h4>
        <div className="space-y-3 text-base">
          <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
            <span className="text-emerald-200 font-semibold">開発言語:</span>
            <span className="text-emerald-100 font-medium">Rust + TypeScript</span>
          </div>
          <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
            <span className="text-emerald-200 font-semibold">ライセンス:</span>
            <span className="text-emerald-100 font-medium">MIT License</span>
          </div>
          <div className="flex justify-between items-center p-3 bg-white/5 rounded-lg">
            <span className="text-emerald-200 font-semibold">最終更新:</span>
            <span className="text-emerald-100 font-medium">2024年12月</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

// Main App Component
const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('compress')
  const [particles] = useState(() => Array.from({ length: 20 }, (_, i) => i))

  const tabs = [
    { id: 'compress', title: '圧縮', icon: <Archive size={20} /> },
    { id: 'extract', title: '展開', icon: <Download size={20} /> },
    { id: 'settings', title: '設定', icon: <Settings size={20} /> },
    { id: 'about', title: '情報', icon: <Info size={20} /> },
  ]

  const renderTabContent = () => {
    switch (activeTab) {
      case 'compress':
        return <CompressionTab />
      case 'extract':
        return <ExtractionTab />
      case 'settings':
        return <SettingsTab />
      case 'about':
        return <AboutTab />
      default:
        return <CompressionTab />
    }
  }

  return (
    <div className="min-h-screen grid-pattern relative overflow-hidden">
      {/* Background Particles */}
      {particles.map((_, index) => (
        <Particle key={index} delay={index * 0.3} />
      ))}

      {/* Main Container */}
      <div className="relative z-10 container mx-auto px-6 py-8 h-screen flex flex-col max-w-7xl">
        {/* Enhanced Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-10"
        >
          <motion.div
            className="relative inline-block"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            <motion.h1
              className="text-5xl md:text-7xl font-bold gradient-text mb-3 relative z-10"
              animate={{ 
                textShadow: [
                  "0 0 20px rgba(16, 185, 129, 0.5)",
                  "0 0 40px rgba(16, 185, 129, 0.7)",
                  "0 0 20px rgba(16, 185, 129, 0.5)"
                ]
              }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              NXZip
            </motion.h1>
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-emerald-600 via-emerald-500 to-emerald-600 rounded-lg blur-2xl opacity-30 -z-10"
              animate={{ 
                scale: [1, 1.1, 1],
                opacity: [0.3, 0.5, 0.3]
              }}
              transition={{ duration: 3, repeat: Infinity }}
            />
          </motion.div>
          <motion.p 
            className="text-xl md:text-2xl text-emerald-200 font-semibold"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            次世代統合アーカイブシステム
          </motion.p>
        </motion.header>

        {/* Enhanced Navigation */}
        <motion.nav
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex justify-center mb-10"
        >
          <motion.div 
            className="glass rounded-3xl p-3 flex gap-3 border border-emerald-400/30 shadow-2xl shadow-emerald-500/20"
            whileHover={{ scale: 1.02 }}
            transition={{ type: "spring", stiffness: 300 }}
          >
            {tabs.map((tab) => (
              <Tab
                key={tab.id}
                id={tab.id}
                title={tab.title}
                icon={tab.icon}
                isActive={activeTab === tab.id}
                onClick={() => setActiveTab(tab.id)}
              />
            ))}
          </motion.div>
        </motion.nav>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3 }}
            >
              {renderTabContent()}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

export default App
