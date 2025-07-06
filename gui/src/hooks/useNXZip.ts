import { invoke } from '@tauri-apps/api/core'
import { open } from '@tauri-apps/plugin-dialog'
import { useState, useCallback } from 'react'

export interface CompressionOptions {
  algorithm: string
  level: number
  encryption?: string
  password?: string
  kdf?: string
}

export interface CompressionResult {
  success: boolean
  message: string
  outputPath?: string
  compressionRatio?: number
  originalSize?: number
  compressedSize?: number
}

export interface ExtractionResult {
  success: boolean
  message: string
  outputPath?: string
  extractedSize?: number
}

export interface FileInfo {
  filename: string
  originalSize: number
  compressedSize: number
  compressionRatio: number
  algorithm: string
  isEncrypted: boolean
  encryptionAlgorithm?: string
  createdAt: string
}

export const useNXZip = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const compressFile = useCallback(async (
    inputPath: string,
    outputPath: string,
    options: CompressionOptions
  ): Promise<CompressionResult | null> => {
    setIsLoading(true)
    setError(null)
    
    try {
      const result = await invoke<CompressionResult>('compress_file', {
        inputPath,
        outputPath,
        options: {
          algorithm: options.algorithm,
          level: options.level,
          encryption: options.encryption || null,
          password: options.password || null,
          kdf: options.kdf || null,
        }
      })
      return result
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const extractFile = useCallback(async (
    inputPath: string,
    outputPath: string,
    password?: string
  ): Promise<ExtractionResult | null> => {
    setIsLoading(true)
    setError(null)
    
    try {
      const result = await invoke<ExtractionResult>('extract_file', {
        inputPath,
        outputPath,
        password: password || null,
      })
      return result
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const getFileInfo = useCallback(async (filePath: string): Promise<FileInfo | null> => {
    setIsLoading(true)
    setError(null)
    
    try {
      const result = await invoke<FileInfo>('get_file_info', { filePath })
      return result
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const selectFile = useCallback(async (filters?: Array<{ name: string; extensions: string[] }>): Promise<string | null> => {
    try {
      const result = await open({
        multiple: false,
        directory: false,
        filters: filters || [
          { name: 'All Files', extensions: ['*'] },
          { name: 'NXZ Archives', extensions: ['nxz', 'nxz.sec'] }
        ]
      })
      return result as string | null
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
      return null
    }
  }, [])

  const selectDirectory = useCallback(async (): Promise<string | null> => {
    try {
      const result = await open({
        multiple: false,
        directory: true,
      })
      return result as string | null
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
      return null
    }
  }, [])

  const selectMultipleFiles = useCallback(async (filters?: Array<{ name: string; extensions: string[] }>): Promise<string[] | null> => {
    try {
      const result = await open({
        multiple: true,
        directory: false,
        filters: filters || [
          { name: 'All Files', extensions: ['*'] }
        ]
      })
      return result as string[] | null
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
      return null
    }
  }, [])

  return {
    isLoading,
    error,
    compressFile,
    extractFile,
    getFileInfo,
    selectFile,
    selectDirectory,
    selectMultipleFiles,
  }
}
