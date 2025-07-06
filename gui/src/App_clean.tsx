import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Archive, 
  Unlock, 
  Settings, 
  Info,
  FolderOpen,
  AlertTriangle,
  CheckCircle2,
  X,
  ChevronDown,
  Play,
  Loader2
} from 'lucide-react'
import { useNXZip } from './hooks/useNXZip'

// ファイルドロップエリアコンポーネント
interface DropZoneProps {
  onFileSelect: () => void
  selectedFiles: string[]
  onRemoveFile: (index: number) => void
  type: 'compress' | 'extract'
  isLoading: boolean
}

const DropZone: React.FC<DropZoneProps> = ({ onFileSelect, selectedFiles, onRemoveFile, type, isLoading }) => {
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    // Tauri handles file drops differently - placeholder
  }

  const icon = type === 'compress' ? <Archive size={48} /> : <Unlock size={48} />
  const title = type === 'compress' ? 'ファイルを圧縮' : 'ファイルを展開'
  const subtitle = type === 'compress' ? 
    'ファイルをここにドラッグ＆ドロップまたはクリックして選択' : 
    '.nxz / .nxz.sec ファイルをここにドラッグ＆ドロップまたはクリックして選択'

  return (
    <div className="space-y-4">
      <motion.div
        className={`
          relative border-2 border-dashed rounded-2xl p-8 transition-all duration-300 cursor-pointer
          ${isDragOver 
            ? 'border-cyan-400 bg-cyan-500/10 scale-[1.02] shadow-lg shadow-cyan-500/20' 
            : 'border-white/30 hover:border-cyan-300 hover:bg-white/5'
          }
          ${isLoading ? 'pointer-events-none opacity-50' : ''}
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={onFileSelect}
        whileHover={{ scale: isLoading ? 1 : 1.01 }}
        whileTap={{ scale: isLoading ? 1 : 0.99 }}
      >
        <div className="text-center">
          <motion.div 
            className={`mx-auto mb-4 w-16 h-16 rounded-2xl flex items-center justify-center transition-all
              ${isDragOver ? 'bg-cyan-500/20 text-cyan-300' : 'bg-white/10 text-white/70'}`}
            whileHover={{ rotate: 5 }}
          >
            {icon}
          </motion.div>
          
          <h3 className="text-xl font-semibold mb-2 text-white">{title}</h3>
          <p className="text-white/60 text-sm mb-6">{subtitle}</p>
          
          <motion.div 
            className="inline-flex items-center px-6 py-3 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-cyan-300 rounded-lg hover:from-cyan-500/30 hover:to-blue-500/30 transition-all border border-cyan-500/30"
            whileHover={{ scale: 1.05 }}
          >
            <FolderOpen size={18} className="mr-2" />
            ファイル選択
          </motion.div>
        </div>
        
        {isLoading && (
          <div className="absolute inset-0 bg-black/20 rounded-2xl flex items-center justify-center">
            <Loader2 className="animate-spin h-8 w-8 text-cyan-400" />
          </div>
        )}
      </motion.div>

      {/* 選択されたファイル一覧 */}
      {selectedFiles.length > 0 && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          className="bg-gradient-to-r from-white/5 to-white/10 rounded-xl p-4 border border-white/10"
        >
          <h4 className="text-sm font-medium text-cyan-300 mb-3 flex items-center">
            <Archive size={16} className="mr-2" />
            選択されたファイル ({selectedFiles.length})
          </h4>
          <div className="space-y-2 max-h-32 overflow-y-auto custom-scrollbar">
            {selectedFiles.map((file, index) => (
              <motion.div 
                key={index} 
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center justify-between text-sm text-white/80 bg-white/5 rounded-lg p-3 hover:bg-white/10 transition-colors"
              >
                <span className="truncate mr-2 font-medium">{file.split('\\').pop() || file.split('/').pop()}</span>
                <button 
                  onClick={(e) => {
                    e.stopPropagation()
                    onRemoveFile(index)
                  }}
                  className="text-white/50 hover:text-red-400 transition-colors hover:scale-110"
                >
                  <X size={16} />
                </button>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}

// タブコンポーネント
interface TabProps {
  id: string
  title: string
  icon: React.ReactNode
  isActive: boolean
  onClick: () => void
}

const Tab: React.FC<TabProps> = ({ title, icon, isActive, onClick }) => (
  <motion.button
    className={`
      flex items-center gap-3 px-6 py-3 rounded-xl transition-all duration-300 font-medium
      ${isActive 
        ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 text-white shadow-lg border border-cyan-500/30' 
        : 'text-white/70 hover:bg-white/10 hover:text-white'
      }
    `}
    onClick={onClick}
    whileHover={{ scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
  >
    <span className="text-lg">{icon}</span>
    <span>{title}</span>
  </motion.button>
)

// 設定セクション
interface SettingsProps {
  options: any
  onChange: (options: any) => void
  type: 'compress' | 'extract'
}

const SettingsSection: React.FC<SettingsProps> = ({ options, onChange, type }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  if (type === 'extract') {
    return (
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-white/5 to-white/10 rounded-xl p-6 border border-white/10"
      >
        <h4 className="font-semibold mb-4 text-cyan-300 flex items-center">
          <Unlock size={18} className="mr-2" />
          展開設定
        </h4>
        <div>
          <label className="block text-sm font-medium text-white/80 mb-2">
            パスワード（暗号化ファイルの場合）
          </label>
          <input
            type="password"
            value={options.password || ''}
            onChange={(e) => onChange({ ...options, password: e.target.value })}
            placeholder="暗号化されている場合はパスワードを入力"
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:border-cyan-400 focus:outline-none transition-colors"
          />
        </div>
      </motion.div>
    )
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-r from-white/5 to-white/10 rounded-xl p-6 border border-white/10"
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between w-full mb-3 text-cyan-300 font-semibold hover:text-cyan-200 transition-colors"
      >
        <div className="flex items-center">
          <Settings size={18} className="mr-2" />
          <span>圧縮設定</span>
        </div>
        <ChevronDown 
          className={`transition-transform ${isExpanded ? 'rotate-180' : ''}`}
          size={20}
        />
      </button>
      
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="space-y-4"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-white/80 mb-2">圧縮アルゴリズム</label>
                <select
                  value={options.algorithm}
                  onChange={(e) => onChange({ ...options, algorithm: e.target.value })}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:border-cyan-400 focus:outline-none appearance-none transition-colors"
                >
                  <option value="auto">自動選択</option>
                  <option value="zstd">Zstd（高速）</option>
                  <option value="lzma2">LZMA2（高圧縮）</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-white/80 mb-2">圧縮レベル</label>
                <select
                  value={options.level}
                  onChange={(e) => onChange({ ...options, level: parseInt(e.target.value) })}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:border-cyan-400 focus:outline-none appearance-none transition-colors"
                >
                  <option value={1}>1（高速）</option>
                  <option value={3}>3（バランス）</option>
                  <option value={6}>6（推奨）</option>
                  <option value={9}>9（最高圧縮）</option>
                </select>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-white/80 mb-2">暗号化</label>
              <select
                value={options.encryption || ''}
                onChange={(e) => onChange({ ...options, encryption: e.target.value })}
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:border-cyan-400 focus:outline-none appearance-none transition-colors"
              >
                <option value="">暗号化なし</option>
                <option value="aes-gcm">AES-256-GCM</option>
                <option value="xchacha20">XChaCha20-Poly1305</option>
              </select>
            </div>
            
            {options.encryption && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <label className="block text-sm font-medium text-white/80 mb-2">パスワード</label>
                <input
                  type="password"
                  value={options.password || ''}
                  onChange={(e) => onChange({ ...options, password: e.target.value })}
                  placeholder="暗号化パスワードを入力"
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-white/50 focus:border-cyan-400 focus:outline-none transition-colors"
                />
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// メインアプリコンポーネント
const App: React.FC = () => {
  const { compressFile, extractFile, selectFile, selectMultipleFiles, isLoading, error } = useNXZip()
  const [activeTab, setActiveTab] = useState('compress')
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [result, setResult] = useState<any>(null)
  const [compressOptions, setCompressOptions] = useState({
    algorithm: 'auto',
    level: 6,
    encryption: '',
    password: '',
    kdf: 'argon2'
  })
  const [extractOptions, setExtractOptions] = useState({
    password: ''
  })

  const handleFileSelect = async () => {
    try {
      setSelectedFiles([])
      setResult(null)
      
      if (activeTab === 'compress') {
        const files = await selectMultipleFiles()
        if (files) {
          setSelectedFiles(files)
        }
      } else {
        const file = await selectFile([
          { name: 'NXZ Archives', extensions: ['nxz', 'nxz.sec'] }
        ])
        if (file) {
          setSelectedFiles([file])
        }
      }
    } catch (err) {
      console.error('File selection error:', err)
    }
  }

  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const handleCompress = async () => {
    if (selectedFiles.length === 0) return

    try {
      const inputPath = selectedFiles[0]
      const outputPath = `${inputPath}.nxz`

      const options = {
        algorithm: compressOptions.algorithm,
        level: compressOptions.level,
        encryption: compressOptions.encryption || undefined,
        password: compressOptions.password || undefined,
        kdf: compressOptions.kdf,
      }

      const compressResult = await compressFile(inputPath, outputPath, options)
      setResult(compressResult)
    } catch (err) {
      console.error('Compression error:', err)
    }
  }

  const handleExtract = async () => {
    if (selectedFiles.length === 0) return

    try {
      const inputPath = selectedFiles[0]
      const fileName = inputPath.split('\\').pop()?.split('.')[0] || 'extracted'
      const outputPath = `${inputPath.replace(/\.[^/.]+$/, '')}_extracted.txt`

      const extractResult = await extractFile(inputPath, outputPath, extractOptions.password)
      setResult(extractResult)
    } catch (err) {
      console.error('Extraction error:', err)
    }
  }

  const tabs = [
    { id: 'compress', title: '圧縮', icon: <Archive size={20} /> },
    { id: 'extract', title: '展開', icon: <Unlock size={20} /> },
    { id: 'settings', title: '設定', icon: <Settings size={20} /> },
    { id: 'info', title: '情報', icon: <Info size={20} /> }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-800 text-white overflow-hidden">
      {/* 背景パーティクル */}
      <div className="absolute inset-0 overflow-hidden">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400/30 rounded-full"
            animate={{
              x: [0, Math.random() * 100 - 50],
              y: [0, Math.random() * 100 - 50],
              opacity: [0, 1, 0]
            }}
            transition={{
              duration: Math.random() * 3 + 2,
              repeat: Infinity,
              delay: i * 0.1
            }}
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 p-6">
        {/* ヘッダー */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <motion.h1 
            className="text-5xl font-bold mb-2"
            style={{
              background: 'linear-gradient(135deg, #00f5ff, #8b5cf6, #06b6d4)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text'
            }}
            whileHover={{ scale: 1.05 }}
          >
            NXZip
          </motion.h1>
          <p className="text-white/70 text-lg">次世代統合アーカイブシステム</p>
        </motion.div>

        {/* タブナビゲーション */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex justify-center mb-8"
        >
          <div className="flex gap-2 bg-white/10 p-2 rounded-2xl backdrop-blur-sm border border-white/20">
            {tabs.map(tab => (
              <Tab
                key={tab.id}
                id={tab.id}
                title={tab.title}
                icon={tab.icon}
                isActive={activeTab === tab.id}
                onClick={() => {
                  setActiveTab(tab.id)
                  setSelectedFiles([])
                  setResult(null)
                }}
              />
            ))}
          </div>
        </motion.div>

        {/* エラー表示 */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto mb-6 p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-300 flex items-center gap-2"
          >
            <AlertTriangle size={20} />
            <span>{error}</span>
          </motion.div>
        )}

        {/* 成功表示 */}
        {result && result.success && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto mb-6 p-4 bg-green-500/20 border border-green-500/50 rounded-xl text-green-300"
          >
            <div className="flex items-center gap-2 mb-2">
              <CheckCircle2 size={20} />
              <span className="font-semibold">{result.message}</span>
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

        {/* メインコンテンツ */}
        <div className="max-w-2xl mx-auto">
          <AnimatePresence mode="wait">
            {activeTab === 'compress' && (
              <motion.div
                key="compress"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-6"
              >
                <DropZone
                  onFileSelect={handleFileSelect}
                  selectedFiles={selectedFiles}
                  onRemoveFile={handleRemoveFile}
                  type="compress"
                  isLoading={isLoading}
                />
                
                <SettingsSection
                  options={compressOptions}
                  onChange={setCompressOptions}
                  type="compress"
                />

                <motion.button
                  onClick={handleCompress}
                  disabled={selectedFiles.length === 0 || isLoading}
                  className="w-full py-4 px-6 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-xl font-semibold
                    disabled:opacity-50 disabled:cursor-not-allowed hover:from-cyan-600 hover:to-blue-600 
                    transition-all duration-300 flex items-center justify-center gap-3 shadow-lg"
                  whileHover={{ scale: selectedFiles.length > 0 && !isLoading ? 1.02 : 1 }}
                  whileTap={{ scale: selectedFiles.length > 0 && !isLoading ? 0.98 : 1 }}
                >
                  {isLoading ? (
                    <Loader2 className="animate-spin" size={20} />
                  ) : (
                    <Play size={20} />
                  )}
                  圧縮開始
                </motion.button>
              </motion.div>
            )}

            {activeTab === 'extract' && (
              <motion.div
                key="extract"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-6"
              >
                <DropZone
                  onFileSelect={handleFileSelect}
                  selectedFiles={selectedFiles}
                  onRemoveFile={handleRemoveFile}
                  type="extract"
                  isLoading={isLoading}
                />
                
                <SettingsSection
                  options={extractOptions}
                  onChange={setExtractOptions}
                  type="extract"
                />

                <motion.button
                  onClick={handleExtract}
                  disabled={selectedFiles.length === 0 || isLoading}
                  className="w-full py-4 px-6 bg-gradient-to-r from-emerald-500 to-cyan-500 rounded-xl font-semibold
                    disabled:opacity-50 disabled:cursor-not-allowed hover:from-emerald-600 hover:to-cyan-600 
                    transition-all duration-300 flex items-center justify-center gap-3 shadow-lg"
                  whileHover={{ scale: selectedFiles.length > 0 && !isLoading ? 1.02 : 1 }}
                  whileTap={{ scale: selectedFiles.length > 0 && !isLoading ? 0.98 : 1 }}
                >
                  {isLoading ? (
                    <Loader2 className="animate-spin" size={20} />
                  ) : (
                    <Unlock size={20} />
                  )}
                  展開開始
                </motion.button>
              </motion.div>
            )}

            {activeTab === 'settings' && (
              <motion.div
                key="settings"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="text-center py-12"
              >
                <motion.div 
                  className="mx-auto mb-6 w-20 h-20 rounded-full bg-gradient-to-r from-cyan-500/20 to-purple-500/20 flex items-center justify-center border border-cyan-500/30"
                  whileHover={{ scale: 1.1, rotate: 180 }}
                >
                  <Settings size={40} className="text-cyan-300" />
                </motion.div>
                <h3 className="text-2xl font-semibold mb-2 text-cyan-300">設定</h3>
                <p className="text-white/60">アプリケーション設定は近日実装予定です</p>
              </motion.div>
            )}

            {activeTab === 'info' && (
              <motion.div
                key="info"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-6"
              >
                <div className="bg-gradient-to-r from-white/5 to-white/10 rounded-xl p-6 text-center border border-white/10">
                  <motion.h3 
                    className="text-3xl font-bold mb-4"
                    style={{
                      background: 'linear-gradient(135deg, #00f5ff, #8b5cf6)',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text'
                    }}
                    whileHover={{ scale: 1.05 }}
                  >
                    NXZip
                  </motion.h3>
                  <p className="text-white/80 mb-6 text-lg">次世代統合アーカイブシステム</p>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <motion.div 
                      className="bg-gradient-to-r from-white/5 to-white/10 rounded-lg p-4 border border-white/10"
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="text-cyan-300 font-semibold">バージョン</div>
                      <div className="text-white">1.0.0</div>
                    </motion.div>
                    <motion.div 
                      className="bg-gradient-to-r from-white/5 to-white/10 rounded-lg p-4 border border-white/10"
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="text-cyan-300 font-semibold">ライセンス</div>
                      <div className="text-white">MIT</div>
                    </motion.div>
                    <motion.div 
                      className="bg-gradient-to-r from-white/5 to-white/10 rounded-lg p-4 border border-white/10"
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="text-cyan-300 font-semibold">対応形式</div>
                      <div className="text-white">.nxz, .nxz.sec</div>
                    </motion.div>
                    <motion.div 
                      className="bg-gradient-to-r from-white/5 to-white/10 rounded-lg p-4 border border-white/10"
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="text-cyan-300 font-semibold">暗号化</div>
                      <div className="text-white">AES-256, XChaCha20</div>
                    </motion.div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-white/5 to-white/10 rounded-xl p-6 border border-white/10">
                  <h4 className="text-lg font-semibold text-cyan-300 mb-4">特徴</h4>
                  <div className="space-y-3 text-sm text-white/80">
                    <div className="flex items-center">
                      <div className="w-2 h-2 bg-cyan-400 rounded-full mr-3"></div>
                      SPE（Structure-Preserving Encryption）による高度な暗号化
                    </div>
                    <div className="flex items-center">
                      <div className="w-2 h-2 bg-purple-400 rounded-full mr-3"></div>
                      LZMA2/Zstdアルゴリズムによる高効率圧縮
                    </div>
                    <div className="flex items-center">
                      <div className="w-2 h-2 bg-blue-400 rounded-full mr-3"></div>
                      多層セキュリティ機構による強固な保護
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

export default App
