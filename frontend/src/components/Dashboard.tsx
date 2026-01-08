import React, { useState } from 'react';
import { Menu, X, Home, Camera, Settings, Bell, AlertCircle, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import CameraGrid from './CameraGrid';

type DetectionMode = 'qr-alerts' | 'anti-cheat' | 'anomaly';
type PageType = 'dashboard' | 'camera-feed';

const Dashboard: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [systemStatus] = useState<'ok' | 'alert'>('ok');
  const [activeMode, setActiveMode] = useState<DetectionMode>('qr-alerts');
  const [activePage, setActivePage] = useState<PageType>('dashboard');

  const modes = [
    { id: 'qr-alerts' as DetectionMode, label: 'QR Alerts' },
    { id: 'anti-cheat' as DetectionMode, label: 'Anti Cheat' },
    { id: 'anomaly' as DetectionMode, label: 'Anomaly Detection' },
  ];

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 fixed top-0 left-0 right-0 z-50">
        <div className="flex items-center justify-between h-16 px-4">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="hover:bg-slate-100"
            >
              {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            <h1 className="text-xl font-semibold text-slate-900">Drishya</h1>
          </div>

          {/* Sliding Toggle */}
          <div className="hidden md:flex items-center">
            <div className="relative bg-slate-100 rounded-lg p-1 flex gap-1">
              {/* Sliding background */}
              <div
                className={cn(
                  "absolute top-1 bottom-1 bg-white rounded-md shadow-sm transition-all duration-300 ease-in-out",
                  activeMode === 'qr-alerts' && "left-1 w-[100px]",
                  activeMode === 'anti-cheat' && "left-[calc(100px+0.25rem+0.25rem)] w-[100px]",
                  activeMode === 'anomaly' && "left-[calc(200px+0.5rem+0.25rem)] w-[160px]"
                )}
              />
              
              {/* Buttons */}
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={cn(
                    "relative z-10 px-4 py-2 text-sm font-medium rounded-md transition-colors duration-200",
                    activeMode === mode.id
                      ? "text-slate-900"
                      : "text-slate-600 hover:text-slate-900"
                  )}
                  style={{
                    width: mode.id === 'anomaly' ? '160px' : '100px'
                  }}
                >
                  {mode.label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="hover:bg-slate-100 relative">
              <Bell className="h-5 w-5" />
              <span className="absolute top-2 right-2 h-2 w-2 bg-red-500 rounded-full"></span>
            </Button>
          </div>
        </div>
      </header>

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed left-0 top-16 bottom-0 bg-white border-r border-slate-200 transition-all duration-300 z-40 overflow-hidden',
          sidebarOpen ? 'w-64' : 'w-0'
        )}
      >
        <nav className={cn(
          'p-4 space-y-2 w-64',
          !sidebarOpen && 'opacity-0'
        )}>
          <button
            onClick={() => setActivePage('dashboard')}
            className={cn(
              "flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-lg whitespace-nowrap w-full transition-colors",
              activePage === 'dashboard' 
                ? "text-white bg-slate-900" 
                : "text-slate-600 hover:bg-slate-100"
            )}
          >
            <Home className="h-5 w-5 flex-shrink-0" />
            <span>Dashboard</span>
          </button>
          <button
            onClick={() => setActivePage('camera-feed')}
            className={cn(
              "flex items-center gap-3 px-4 py-3 text-sm font-medium rounded-lg whitespace-nowrap w-full transition-colors",
              activePage === 'camera-feed'
                ? "text-white bg-slate-900"
                : "text-slate-600 hover:bg-slate-100"
            )}
          >
            <Camera className="h-5 w-5 flex-shrink-0" />
            <span>Camera Feed</span>
          </button>
          <a
            href="#"
            className="flex items-center gap-3 px-4 py-3 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors whitespace-nowrap"
          >
            <AlertCircle className="h-5 w-5 flex-shrink-0" />
            <span>Alerts</span>
          </a>
          <a
            href="#"
            className="flex items-center gap-3 px-4 py-3 text-sm font-medium text-slate-600 hover:bg-slate-100 rounded-lg transition-colors whitespace-nowrap"
          >
            <Settings className="h-5 w-5 flex-shrink-0" />
            <span>Settings</span>
          </a>
        </nav>
      </aside>

      {/* Main Content */}
      <main
        className={cn(
          'pt-16 transition-all duration-300',
          sidebarOpen ? 'ml-64' : 'ml-0'
        )}
      >
        <div className="p-6">
          {/* Dashboard Header with Status */}
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-semibold text-slate-900">
              {activeMode === 'qr-alerts' && 'QR Alerts Dashboard'}
              {activeMode === 'anti-cheat' && 'Anti Cheat Monitoring'}
              {activeMode === 'anomaly' && 'Anomaly Detection'}
            </h2>
            <div className="flex items-center gap-2">
              {systemStatus === 'ok' ? (
                <>
                  <CheckCircle className="h-6 w-6 text-green-600" />
                  <span className="text-sm font-medium text-green-600">All Systems Operational</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-6 w-6 text-red-600" />
                  <span className="text-sm font-medium text-red-600">Alert Detected</span>
                </>
              )}
            </div>
          </div>

          {/* Content based on active page */}
          {activePage === 'dashboard' && (
            <>
              {/* Camera Grid based on active mode */}
              {activeMode === 'qr-alerts' && <CameraGrid mode="qr" />}
              {activeMode === 'anti-cheat' && <CameraGrid mode="anticheat" />}
              {activeMode === 'anomaly' && <CameraGrid mode="anomaly" />}
            </>
          )}

          {activePage === 'camera-feed' && (
            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
              <h3 className="text-lg font-semibold text-slate-900 mb-4">Raw Camera Feed</h3>
              <div className="relative bg-slate-900 rounded-lg overflow-hidden aspect-video">
                <img
                  src="http://localhost:5004/api/raw_feed"
                  alt="Raw camera stream"
                  className="w-full h-full object-contain"
                />
              </div>
              <div className="mt-4 text-sm text-slate-600">
                <p>This is the unprocessed camera stream from Redis without any detection overlays.</p>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
