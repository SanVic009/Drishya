import React, { useState, useEffect } from 'react';
import { X, Maximize2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import VideoSelector from './VideoSelector';

interface CameraFeed {
  id: number;
  name: string;
  thumbnail: string;
  location: string;
  status: 'active' | 'inactive';
  streamUrl?: string;
  isLive?: boolean;
  isVideoSelector?: boolean;
}

interface CameraGridProps {
  mode: 'qr' | 'anticheat' | 'anomaly';
}

const CameraGrid: React.FC<CameraGridProps> = ({ mode }) => {
  const [selectedCamera, setSelectedCamera] = useState<CameraFeed | null>(null);
  const [isProcessingClass3, setIsProcessingClass3] = useState(false);

  // Start cheating.mp4 processing for Class 3 when component mounts in anticheat mode
  useEffect(() => {
    if (mode === 'anticheat') {
      startClass3Video();
    }
  }, [mode]);

  const startClass3Video = async () => {
    try {
      setIsProcessingClass3(true);
      const response = await fetch('http://localhost:5002/api/videos/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: 'cheating.mp4' }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to process video');
      }
      console.log('Class 3 video processing started: cheating.mp4');
    } catch (err: any) {
      console.error('Error starting Class 3 video:', err);
      setIsProcessingClass3(false);
    }
  };

  // Configure camera feeds based on mode
  const getCameraFeeds = (): CameraFeed[] => {
    if (mode === 'qr') {
      return [
        {
          id: 1,
          name: 'Camera 01',
          thumbnail: 'http://localhost:5003/api/qr_feed',
          location: 'Main Entrance',
          status: 'active',
          streamUrl: 'http://localhost:5003/api/qr_feed',
          isLive: true,
        },
        {
          id: 2,
          name: 'Camera 02',
          thumbnail: 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600&h=400&fit=crop',
          location: 'Parking Lot',
          status: 'active',
          isLive: false,
        },
        {
          id: 3,
          name: 'Camera 03',
          thumbnail: 'https://images.unsplash.com/photo-1497366216548-37526070297c?w=600&h=400&fit=crop',
          location: 'Office Floor',
          status: 'active',
          isLive: false,
        },
        {
          id: 4,
          name: 'Camera 04',
          thumbnail: 'https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=600&h=400&fit=crop',
          location: 'Warehouse',
          status: 'active',
          isLive: false,
        },
        {
          id: 5,
          name: 'Camera 05',
          thumbnail: 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600&h=400&fit=crop',
          location: 'Loading Bay',
          status: 'active',
          isLive: false,
        },
        {
          id: 6,
          name: 'Camera 06',
          thumbnail: 'https://images.unsplash.com/photo-1557597774-9d273605dfa9?w=600&h=400&fit=crop',
          location: 'Emergency Exit',
          status: 'active',
          isLive: false,
        },
      ];
    } else if (mode === 'anticheat') {
      return [
        {
          id: 1,
          name: 'Class 1',
          thumbnail: 'http://localhost:5002/api/anticheat_feed',
          location: 'Room 101',
          status: 'active',
          streamUrl: 'http://localhost:5002/api/anticheat_feed',
          isLive: true,
        },
        {
          id: 2,
          name: 'Class 2',
          thumbnail: '',
          location: 'Test Videos',
          status: 'active',
          isLive: false,
          isVideoSelector: true, // Special flag for video selector
        },
        {
          id: 3,
          name: 'Class 3',
          thumbnail: 'http://localhost:5002/api/video_feed',
          location: 'Room 103 - cheating.mp4',
          status: 'active',
          streamUrl: 'http://localhost:5002/api/video_feed',
          isLive: true,
        },
        {
          id: 4,
          name: 'Class 4',
          thumbnail: 'https://images.unsplash.com/photo-1562774053-701939374585?w=600&h=400&fit=crop',
          location: 'Room 201',
          status: 'active',
          isLive: false,
        },
        {
          id: 5,
          name: 'Class 5',
          thumbnail: 'https://images.unsplash.com/photo-1509062522246-3755977927d7?w=600&h=400&fit=crop',
          location: 'Room 202',
          status: 'active',
          isLive: false,
        },
        {
          id: 6,
          name: 'Class 6',
          thumbnail: 'https://images.unsplash.com/photo-1523580494863-6f3031224c94?w=600&h=400&fit=crop',
          location: 'Room 203',
          status: 'active',
          isLive: false,
        },
      ];
    } else { // anomaly mode
      return [
        {
          id: 1,
          name: 'Camera 01',
          thumbnail: 'http://localhost:5001/api/video_feed',
          location: 'Main Entrance',
          status: 'active',
          streamUrl: 'http://localhost:5001/api/video_feed',
          isLive: true,
        },
        {
          id: 2,
          name: 'Camera 02',
          thumbnail: 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600&h=400&fit=crop',
          location: 'Parking Lot',
          status: 'active',
          isLive: false,
        },
        {
          id: 3,
          name: 'Camera 03',
          thumbnail: 'https://images.unsplash.com/photo-1497366216548-37526070297c?w=600&h=400&fit=crop',
          location: 'Office Floor',
          status: 'active',
          isLive: false,
        },
        {
          id: 4,
          name: 'Camera 04',
          thumbnail: 'https://images.unsplash.com/photo-1497366811353-6870744d04b2?w=600&h=400&fit=crop',
          location: 'Warehouse',
          status: 'active',
          isLive: false,
        },
        {
          id: 5,
          name: 'Camera 05',
          thumbnail: 'https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=600&h=400&fit=crop',
          location: 'Loading Bay',
          status: 'active',
          isLive: false,
        },
        {
          id: 6,
          name: 'Camera 06',
          thumbnail: 'https://images.unsplash.com/photo-1557597774-9d273605dfa9?w=600&h=400&fit=crop',
          location: 'Emergency Exit',
          status: 'active',
          isLive: false,
        },
      ];
    }
  };

  const cameraFeeds = getCameraFeeds();

  // Check if we should show video selector for Class 2 in anticheat mode
  const showVideoSelector = mode === 'anticheat' && cameraFeeds.some(c => c.id === 2 && c.isVideoSelector);

  return (
    <>
      {/* Video Selector for Class 2 (Anti-cheat mode only) */}
      {showVideoSelector && <VideoSelector />}
      
      {/* Camera Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {cameraFeeds.filter(c => !c.isVideoSelector).map((camera) => (
          <div
            key={camera.id}
            className="group relative bg-white rounded-lg border border-slate-200 overflow-hidden hover:shadow-lg transition-all duration-300 cursor-pointer"
            onClick={() => setSelectedCamera(camera)}
          >
            {/* Camera Thumbnail */}
            <div className="relative aspect-video bg-slate-900 overflow-hidden">
              <img
                src={camera.thumbnail}
                alt={camera.name}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
              
              {/* Overlay on hover */}
              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                <Maximize2 className="h-8 w-8 text-white" />
              </div>

              {/* Status Indicator */}
              <div className="absolute top-3 right-3 flex items-center gap-2">
                <div className={cn(
                  "h-2 w-2 rounded-full",
                  camera.status === 'active' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                )}></div>
                <span className="text-xs font-medium text-white bg-black/50 px-2 py-1 rounded">
                  {camera.isLive ? 'LIVE STREAM' : camera.status === 'active' ? 'LIVE' : 'OFFLINE'}
                </span>
              </div>
            </div>

            {/* Camera Info */}
            <div className="p-4">
              <h3 className="font-semibold text-slate-900 mb-1">{camera.name}</h3>
              <p className="text-sm text-slate-600">{camera.location}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Modal for Focused Camera View */}
      {selectedCamera && (
        <div
          className="fixed inset-0 bg-black/80 z-[100] flex items-center justify-center p-4"
          onClick={() => setSelectedCamera(null)}
        >
          <div
            className="relative bg-white rounded-lg max-w-5xl w-full overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close Button */}
            <button
              onClick={() => setSelectedCamera(null)}
              className="absolute top-4 right-4 z-10 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors"
            >
              <X className="h-6 w-6" />
            </button>

            {/* Camera Feed */}
            <div className="relative aspect-video bg-slate-900">
              {selectedCamera.isLive ? (
                <img
                  src={selectedCamera.streamUrl || selectedCamera.thumbnail}
                  alt={selectedCamera.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <img
                  src={selectedCamera.thumbnail}
                  alt={selectedCamera.name}
                  className="w-full h-full object-cover"
                />
              )}

              {/* Status Indicator */}
              <div className="absolute top-4 left-4 flex items-center gap-2">
                <div className={cn(
                  "h-3 w-3 rounded-full",
                  selectedCamera.status === 'active' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                )}></div>
                <span className="text-sm font-medium text-white bg-black/50 px-3 py-1.5 rounded">
                  {selectedCamera.isLive ? 'LIVE STREAM' : selectedCamera.status === 'active' ? 'LIVE' : 'OFFLINE'}
                </span>
              </div>
            </div>

            {/* Camera Details */}
            <div className="p-6 border-t border-slate-200">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold text-slate-900 mb-1">
                    {selectedCamera.name}
                  </h3>
                  <p className="text-slate-600">{selectedCamera.location}</p>
                </div>
                <div className="flex gap-2">
                  <button className="px-4 py-2 text-sm font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors">
                    Take Snapshot
                  </button>
                  <button className="px-4 py-2 text-sm font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors">
                    Record
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default CameraGrid;
