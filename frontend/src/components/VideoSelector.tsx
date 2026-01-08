import React, { useState, useEffect } from 'react';
import { Play, StopCircle, Video, FileVideo } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

interface Video {
  filename: string;
  path: string;
  size: number;
  size_mb: number;
}

const VideoSelector: React.FC = () => {
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch available videos
  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:5002/api/videos/list');
      const data = await response.json();
      
      if (data.videos) {
        setVideos(data.videos);
        if (data.videos.length > 0) {
          setSelectedVideo(data.videos[0].filename);
        }
      }
      setError(null);
    } catch (err) {
      console.error('Error fetching videos:', err);
      setError('Failed to load videos');
    } finally {
      setLoading(false);
    }
  };

  const handleProcessVideo = async () => {
    if (!selectedVideo) {
      return;
    }

    try {
      setIsProcessing(true);
      const response = await fetch('http://localhost:5002/api/videos/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: selectedVideo }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to process video');
      }
    } catch (err: any) {
      console.error('Error processing video:', err);
      alert(`Error: ${err.message}`);
      setIsProcessing(false);
    }
  };

  const handleStopVideo = async () => {
    try {
      await fetch('http://localhost:5002/api/videos/stop', {
        method: 'POST',
      });
      setIsProcessing(false);
    } catch (err) {
      console.error('Error stopping video:', err);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Video Control Panel */}
      <Card className="lg:col-span-1">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileVideo className="h-5 w-5" />
            Test Video Selection
          </CardTitle>
          <CardDescription>
            Select a test video to run anti-cheat detection
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {loading ? (
            <div className="text-center py-4 text-slate-500">
              Loading videos...
            </div>
          ) : error ? (
            <div className="text-center py-4 text-red-500">
              {error}
            </div>
          ) : videos.length === 0 ? (
            <div className="text-center py-4 text-slate-500">
              No videos found
            </div>
          ) : (
            <>
              {/* Video Selector */}
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-700">
                  Select Video
                </label>
                <Select
                  value={selectedVideo}
                  onValueChange={setSelectedVideo}
                  disabled={isProcessing}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Choose a video" />
                  </SelectTrigger>
                  <SelectContent>
                    {videos.map((video) => (
                      <SelectItem key={video.filename} value={video.filename}>
                        <div className="flex items-center justify-between gap-4 w-full">
                          <span className="font-medium">{video.filename}</span>
                          <span className="text-xs text-slate-500">
                            {video.size_mb} MB
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Video Details */}
              {selectedVideo && (
                <div className="bg-slate-50 rounded-lg p-4 space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    <Video className="h-4 w-4 text-slate-500" />
                    <span className="font-medium text-slate-700">
                      {selectedVideo}
                    </span>
                  </div>
                  <div className="text-xs text-slate-500">
                    Size: {videos.find(v => v.filename === selectedVideo)?.size_mb} MB
                  </div>
                </div>
              )}

              {/* Control Buttons */}
              <div className="flex gap-2">
                {!isProcessing ? (
                  <Button
                    onClick={handleProcessVideo}
                    disabled={!selectedVideo}
                    className="flex-1 bg-blue-600 hover:bg-blue-700"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Run Detection
                  </Button>
                ) : (
                  <Button
                    onClick={handleStopVideo}
                    variant="destructive"
                    className="flex-1"
                  >
                    <StopCircle className="h-4 w-4 mr-2" />
                    Stop
                  </Button>
                )}
              </div>

              {/* Processing Status */}
              {isProcessing && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <div className="flex items-center gap-2 text-sm text-blue-700">
                    <div className="h-2 w-2 bg-blue-600 rounded-full animate-pulse"></div>
                    <span>Processing video...</span>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Video Stream Display */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Detection Stream</CardTitle>
          <CardDescription>
            Real-time anti-cheat detection from selected video
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative bg-slate-900 rounded-lg overflow-hidden aspect-video">
            {isProcessing ? (
              <img
                src="http://localhost:5002/api/video_feed"
                alt="Video stream"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400">
                <div className="text-center space-y-2">
                  <Video className="h-16 w-16 mx-auto opacity-50" />
                  <p className="text-sm">Select a video and click "Run Detection"</p>
                </div>
              </div>
            )}
          </div>

          {/* Detection Stats */}
          {isProcessing && (
            <div className="mt-4 grid grid-cols-3 gap-4">
              <div className="bg-slate-50 rounded-lg p-3">
                <div className="text-xs text-slate-500 mb-1">Status</div>
                <div className="text-sm font-semibold text-green-600">
                  Processing
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <div className="text-xs text-slate-500 mb-1">Video</div>
                <div className="text-sm font-semibold text-slate-900 truncate">
                  {selectedVideo}
                </div>
              </div>
              <div className="bg-slate-50 rounded-lg p-3">
                <div className="text-xs text-slate-500 mb-1">Mode</div>
                <div className="text-sm font-semibold text-blue-600">
                  Anti-Cheat
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default VideoSelector;
