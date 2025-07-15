import React from 'react'
import { motion } from 'framer-motion'
import { X, Pause, Play, AlertCircle } from 'lucide-react'

export interface ProgressState {
  isActive: boolean
  progress: number
  message: string
  stage?: string
  speed?: string
  timeRemaining?: string
  canCancel?: boolean
  canPause?: boolean
  isPaused?: boolean
}

interface ProgressPanelProps {
  progressState: ProgressState
  onCancel?: () => void
  onPause?: () => void
  onResume?: () => void
  className?: string
}

export const ProgressPanel: React.FC<ProgressPanelProps> = ({
  progressState,
  onCancel,
  onPause,
  onResume,
  className = ''
}) => {
  if (!progressState.isActive) {
    return null
  }

  const { 
    progress, 
    message, 
    stage, 
    speed, 
    timeRemaining, 
    canCancel = true, 
    canPause = false, 
    isPaused = false 
  } = progressState

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`glass rounded-xl p-6 border-blue-400/30 ${className}`}
    >
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse" />
            <h4 className="font-semibold text-blue-100">{message}</h4>
          </div>
          
          <div className="flex items-center gap-2">
            {/* Pause/Resume button */}
            {canPause && (
              <motion.button
                onClick={isPaused ? onResume : onPause}
                className="p-2 rounded-full hover:bg-white/10 text-white/80 hover:text-white transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                title={isPaused ? '再開' : '一時停止'}
              >
                {isPaused ? <Play size={16} /> : <Pause size={16} />}
              </motion.button>
            )}
            
            {/* Cancel button */}
            {canCancel && (
              <motion.button
                onClick={onCancel}
                className="p-2 rounded-full hover:bg-red-500/20 text-red-300 hover:text-red-200 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                title="キャンセル"
              >
                <X size={16} />
              </motion.button>
            )}
          </div>
        </div>

        {/* Stage information */}
        {stage && (
          <div className="text-sm text-white/70">
            <span className="font-medium">現在の処理: </span>
            {stage}
          </div>
        )}

        {/* Progress bar */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-white/80">進行状況</span>
            <span className="text-blue-300 font-medium">{Math.round(progress)}%</span>
          </div>
          
          <div className="progress-bar relative overflow-hidden">
            <motion.div
              className={`progress-fill relative ${isPaused ? 'opacity-50' : ''}`}
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
            >
              {/* Animated shimmer effect */}
              {!isPaused && (
                <motion.div
                  className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent"
                  animate={{
                    x: ['-100%', '100%'],
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: 'linear',
                  }}
                />
              )}
            </motion.div>
          </div>
        </div>

        {/* Additional information */}
        {(speed || timeRemaining) && (
          <div className="flex items-center justify-between text-xs text-white/60">
            {speed && (
              <div className="flex items-center gap-1">
                <span>速度:</span>
                <span className="text-white/80 font-medium">{speed}</span>
              </div>
            )}
            
            {timeRemaining && (
              <div className="flex items-center gap-1">
                <span>残り時間:</span>
                <span className="text-white/80 font-medium">{timeRemaining}</span>
              </div>
            )}
          </div>
        )}

        {/* Paused state indicator */}
        {isPaused && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-lg"
          >
            <Pause size={16} className="text-yellow-300" />
            <span className="text-yellow-200 text-sm font-medium">処理が一時停止されています</span>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}

// Hook for managing progress state
export const useProgress = () => {
  const [progressState, setProgressState] = React.useState<ProgressState>({
    isActive: false,
    progress: 0,
    message: '',
  })

  const startProgress = React.useCallback((message: string, options?: Partial<ProgressState>) => {
    setProgressState({
      isActive: true,
      progress: 0,
      message,
      canCancel: true,
      canPause: false,
      isPaused: false,
      ...options,
    })
  }, [])

  const updateProgress = React.useCallback((progress: number, updates?: Partial<ProgressState>) => {
    setProgressState(prev => ({
      ...prev,
      progress: Math.max(0, Math.min(100, progress)),
      ...updates,
    }))
  }, [])

  const finishProgress = React.useCallback(() => {
    setProgressState(prev => ({
      ...prev,
      isActive: false,
      progress: 100,
    }))
  }, [])

  const pauseProgress = React.useCallback(() => {
    setProgressState(prev => ({
      ...prev,
      isPaused: true,
    }))
  }, [])

  const resumeProgress = React.useCallback(() => {
    setProgressState(prev => ({
      ...prev,
      isPaused: false,
    }))
  }, [])

  const cancelProgress = React.useCallback(() => {
    setProgressState({
      isActive: false,
      progress: 0,
      message: '',
    })
  }, [])

  return {
    progressState,
    startProgress,
    updateProgress,
    finishProgress,
    pauseProgress,
    resumeProgress,
    cancelProgress,
  }
}