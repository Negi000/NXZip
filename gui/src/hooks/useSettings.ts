import { useState, useEffect, useCallback } from 'react'

export interface AppSettings {
  compression: {
    defaultAlgorithm: string
    defaultLevel: number
    defaultEncryption: string
    defaultKdf: string
  }
  performance: {
    threads: string
    memoryUsage: string
  }
  security: {
    defaultEncryption: string
    defaultKdf: string
  }
  ui: {
    theme: string
    showProgress: boolean
    autoSave: boolean
  }
}

const DEFAULT_SETTINGS: AppSettings = {
  compression: {
    defaultAlgorithm: 'auto',
    defaultLevel: 6,
    defaultEncryption: '',
    defaultKdf: 'argon2',
  },
  performance: {
    threads: 'auto',
    memoryUsage: 'medium',
  },
  security: {
    defaultEncryption: 'aes-gcm',
    defaultKdf: 'argon2',
  },
  ui: {
    theme: 'dark',
    showProgress: true,
    autoSave: true,
  },
}

const SETTINGS_KEY = 'nxzip-settings'

export const useSettings = () => {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS)
  const [isLoading, setIsLoading] = useState(true)

  // Load settings from localStorage on mount
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem(SETTINGS_KEY)
      if (savedSettings) {
        const parsed = JSON.parse(savedSettings)
        setSettings({ ...DEFAULT_SETTINGS, ...parsed })
      }
    } catch (error) {
      console.warn('Failed to load settings:', error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Save settings to localStorage
  const saveSettings = useCallback((newSettings: Partial<AppSettings>) => {
    const updatedSettings = { ...settings, ...newSettings }
    setSettings(updatedSettings)
    
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(updatedSettings))
    } catch (error) {
      console.error('Failed to save settings:', error)
    }
  }, [settings])

  // Update specific setting category
  const updateCompressionSettings = useCallback((compression: Partial<AppSettings['compression']>) => {
    saveSettings({ compression: { ...settings.compression, ...compression } })
  }, [settings.compression, saveSettings])

  const updatePerformanceSettings = useCallback((performance: Partial<AppSettings['performance']>) => {
    saveSettings({ performance: { ...settings.performance, ...performance } })
  }, [settings.performance, saveSettings])

  const updateSecuritySettings = useCallback((security: Partial<AppSettings['security']>) => {
    saveSettings({ security: { ...settings.security, ...security } })
  }, [settings.security, saveSettings])

  const updateUISettings = useCallback((ui: Partial<AppSettings['ui']>) => {
    saveSettings({ ui: { ...settings.ui, ...ui } })
  }, [settings.ui, saveSettings])

  // Reset to defaults
  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS)
    try {
      localStorage.removeItem(SETTINGS_KEY)
    } catch (error) {
      console.error('Failed to reset settings:', error)
    }
  }, [])

  return {
    settings,
    isLoading,
    saveSettings,
    updateCompressionSettings,
    updatePerformanceSettings,
    updateSecuritySettings,
    updateUISettings,
    resetSettings,
  }
}