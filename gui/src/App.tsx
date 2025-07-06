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

const Tab: React.FC<TabProps> = ({ title, icon, isActive, onClick }) => (
  <motion.button
    className={`flex items-center gap-3 px-6 py-3 rounded-xl transition-all duration-300 ${
      isActive ? 'glass neon-glow' : 'glass-dark hover:bg-white/10'
    }`}
    onClick={onClick}
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
  >
    <span className="text-xl">{icon}</span>
    <span className="font-medium">{title}</span>
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

      {/* File Selection Area */}
      <div
        className="relative p-12 border-2 border-dashed rounded-2xl transition-all duration-300 border-white/30 glass cursor-pointer hover:border-blue-400"
        onClick={handleFileSelect}
      >
        <div className="text-center">
          <motion.div
            className="floating mx-auto mb-4 w-20 h-20 flex items-center justify-center glass rounded-full"
            whileHover={{ scale: 1.1 }}
          >
            <Upload size={32} className="text-blue-300" />
          </motion.div>
          <h3 className="text-xl font-semibold mb-2">ファイル選択</h3>
          <p className="text-white/70 mb-4">クリックしてファイルを選択</p>
          <div className="btn-primary inline-flex items-center">
            <FolderOpen size={18} className="mr-2" />
            ファイル選択
          </div>
        </div>
      </div>

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

      {/* File Selection */}
      <div 
        className="glass rounded-xl p-12 text-center cursor-pointer hover:bg-white/5 transition-all"
        onClick={handleFileSelect}
      >
        <motion.div
          className="floating mx-auto mb-4 w-20 h-20 flex items-center justify-center glass rounded-full"
          whileHover={{ scale: 1.1 }}
        >
          <Download size={32} className="text-green-300" />
        </motion.div>
        <h3 className="text-xl font-semibold mb-2">アーカイブ展開</h3>
        <p className="text-white/70 mb-4">.nxz または .nxz.sec ファイルを選択してください</p>
        {selectedFile ? (
          <div className="glass rounded-lg p-3 mb-4">
            <span className="text-sm text-green-300">選択済み: </span>
            <span className="text-sm">{selectedFile.split('\\').pop() || selectedFile.split('/').pop()}</span>
          </div>
        ) : null}
        <div className="btn-primary inline-flex items-center">
          <FolderOpen size={18} className="mr-2" />
          {selectedFile ? 'ファイル変更' : 'アーカイブ選択'}
        </div>
      </div>

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

// About Tab Component
const AboutTab: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="glass rounded-xl p-8 text-center">
        <motion.div
          className="floating mx-auto mb-6 w-24 h-24 flex items-center justify-center glass rounded-full neon-glow"
          whileHover={{ scale: 1.1 }}
        >
          <Star size={40} className="gradient-text" />
        </motion.div>
        <h2 className="text-3xl font-bold mb-2 gradient-text">NXZip</h2>
        <p className="text-xl text-white/80 mb-4">次世代統合アーカイブシステム</p>
        <p className="text-white/60">Version 1.0.0</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card text-center">
          <Archive size={32} className="mx-auto mb-3 text-blue-300" />
          <h4 className="font-semibold mb-2">SPE暗号化</h4>
          <p className="text-sm text-white/70">構造保持型暗号化技術</p>
        </div>
        <div className="card text-center">
          <Zap size={32} className="mx-auto mb-3 text-yellow-300" />
          <h4 className="font-semibold mb-2">高速圧縮</h4>
          <p className="text-sm text-white/70">LZMA2/Zstd最適化</p>
        </div>
        <div className="card text-center">
          <Shield size={32} className="mx-auto mb-3 text-green-300" />
          <h4 className="font-semibold mb-2">多層セキュリティ</h4>
          <p className="text-sm text-white/70">AES-GCM/XChaCha20</p>
        </div>
      </div>

      <div className="glass rounded-xl p-6">
        <h4 className="font-semibold mb-4 flex items-center">
          <HardDrive size={18} className="mr-2" />
          システム情報
        </h4>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-white/70">Rust バージョン:</span>
            <span className="ml-2">1.70.0+</span>
          </div>
          <div>
            <span className="text-white/70">GUI フレームワーク:</span>
            <span className="ml-2">Tauri + React</span>
          </div>
          <div>
            <span className="text-white/70">圧縮エンジン:</span>
            <span className="ml-2">Custom NXZ</span>
          </div>
          <div>
            <span className="text-white/70">暗号化:</span>
            <span className="ml-2">AES-256, XChaCha20</span>
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
      <div className="relative z-10 container mx-auto px-6 py-8 h-screen flex flex-col max-w-6xl">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <motion.h1
            className="text-4xl md:text-6xl font-bold gradient-text mb-2"
            whileHover={{ scale: 1.05 }}
          >
            NXZip
          </motion.h1>
          <p className="text-lg text-white/80">次世代統合アーカイブシステム</p>
        </motion.header>

        {/* Navigation */}
        <motion.nav
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex justify-center mb-8"
        >
          <div className="glass rounded-2xl p-2 flex gap-2">
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
          </div>
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
