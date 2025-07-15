import { useState, useCallback, useRef } from 'react'

export interface DragAndDropOptions {
  onFilesDropped: (files: string[]) => void
  acceptedExtensions?: string[]
  multiple?: boolean
}

export const useDragAndDrop = ({ onFilesDropped, acceptedExtensions, multiple = true }: DragAndDropOptions) => {
  const [isDragging, setIsDragging] = useState(false)
  const [dragCounter, setDragCounter] = useState(0)
  const dragRef = useRef<HTMLDivElement>(null)

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragCounter(count => count + 1)
    if (e.dataTransfer?.items && e.dataTransfer.items.length > 0) {
      setIsDragging(true)
    }
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragCounter(count => {
      const newCount = count - 1
      if (newCount === 0) {
        setIsDragging(false)
      }
      return newCount
    })
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
    setDragCounter(0)

    if (e.dataTransfer?.files) {
      const files = Array.from(e.dataTransfer.files)
      const validFiles: string[] = []

      files.forEach(file => {
        if (acceptedExtensions) {
          const extension = file.name.split('.').pop()?.toLowerCase()
          if (extension && acceptedExtensions.includes(extension)) {
            // In a real Tauri app, we'd get the file path
            // For now, we'll use the file name as a placeholder
            validFiles.push(file.name)
          }
        } else {
          validFiles.push(file.name)
        }
      })

      if (!multiple && validFiles.length > 1) {
        validFiles.splice(1)
      }

      if (validFiles.length > 0) {
        onFilesDropped(validFiles)
      }
    }
  }, [onFilesDropped, acceptedExtensions, multiple])

  const dragProps = {
    ref: dragRef,
    onDragEnter: handleDragEnter,
    onDragLeave: handleDragLeave,
    onDragOver: handleDragOver,
    onDrop: handleDrop,
  }

  return {
    isDragging,
    dragProps,
  }
}