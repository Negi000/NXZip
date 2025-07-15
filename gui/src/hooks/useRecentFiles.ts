import { useState, useEffect, useCallback } from 'react'

export interface RecentFile {
  id: string
  fileName: string
  filePath: string
  operation: 'compress' | 'extract'
  timestamp: number
  fileSize?: number
  compressionRatio?: number
  success: boolean
}

const RECENT_FILES_KEY = 'nxzip-recent-files'
const MAX_RECENT_FILES = 20

export const useRecentFiles = () => {
  const [recentFiles, setRecentFiles] = useState<RecentFile[]>([])

  // Load recent files from localStorage on mount
  useEffect(() => {
    try {
      const savedFiles = localStorage.getItem(RECENT_FILES_KEY)
      if (savedFiles) {
        const parsed = JSON.parse(savedFiles)
        setRecentFiles(parsed)
      }
    } catch (error) {
      console.warn('Failed to load recent files:', error)
    }
  }, [])

  // Save recent files to localStorage
  const saveRecentFiles = useCallback((files: RecentFile[]) => {
    try {
      localStorage.setItem(RECENT_FILES_KEY, JSON.stringify(files))
      setRecentFiles(files)
    } catch (error) {
      console.error('Failed to save recent files:', error)
    }
  }, [])

  // Add a new recent file
  const addRecentFile = useCallback((file: Omit<RecentFile, 'id' | 'timestamp'>) => {
    const newFile: RecentFile = {
      ...file,
      id: Date.now().toString(),
      timestamp: Date.now(),
    }

    setRecentFiles(prevFiles => {
      // Remove duplicate entries for the same file path
      const filteredFiles = prevFiles.filter(f => f.filePath !== file.filePath)
      
      // Add new file to the beginning
      const updatedFiles = [newFile, ...filteredFiles]
      
      // Limit to MAX_RECENT_FILES
      const limitedFiles = updatedFiles.slice(0, MAX_RECENT_FILES)
      
      // Save to localStorage
      try {
        localStorage.setItem(RECENT_FILES_KEY, JSON.stringify(limitedFiles))
      } catch (error) {
        console.error('Failed to save recent files:', error)
      }
      
      return limitedFiles
    })
  }, [])

  // Remove a specific recent file
  const removeRecentFile = useCallback((id: string) => {
    setRecentFiles(prevFiles => {
      const updatedFiles = prevFiles.filter(f => f.id !== id)
      saveRecentFiles(updatedFiles)
      return updatedFiles
    })
  }, [saveRecentFiles])

  // Clear all recent files
  const clearRecentFiles = useCallback(() => {
    setRecentFiles([])
    try {
      localStorage.removeItem(RECENT_FILES_KEY)
    } catch (error) {
      console.error('Failed to clear recent files:', error)
    }
  }, [])

  // Get recent files by operation type
  const getRecentFilesByOperation = useCallback((operation: 'compress' | 'extract') => {
    return recentFiles.filter(f => f.operation === operation)
  }, [recentFiles])

  // Get successful operations only
  const getSuccessfulOperations = useCallback(() => {
    return recentFiles.filter(f => f.success)
  }, [recentFiles])

  return {
    recentFiles,
    addRecentFile,
    removeRecentFile,
    clearRecentFiles,
    getRecentFilesByOperation,
    getSuccessfulOperations,
  }
}