import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/', (req, res) => {
  res.json({ message: 'Drishya Backend API is running' });
});

app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'ok',
    timestamp: new Date().toISOString(),
    service: 'Drishya Non-Compliance Detection API'
  });
});

// Camera feeds endpoint (placeholder)
app.get('/api/cameras', (req, res) => {
  res.json({
    cameras: [
      {
        id: 1,
        name: 'Camera 01',
        location: 'Main Entrance',
        status: 'active',
        streamUrl: '/stream/camera-01'
      },
      {
        id: 2,
        name: 'Camera 02',
        location: 'Parking Lot',
        status: 'active',
        streamUrl: '/stream/camera-02'
      },
      {
        id: 3,
        name: 'Camera 03',
        location: 'Office Floor',
        status: 'active',
        streamUrl: '/stream/camera-03'
      }
    ]
  });
});

// Alerts endpoint (placeholder)
app.get('/api/alerts', (req, res) => {
  res.json({
    alerts: [],
    systemStatus: 'ok'
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Drishya Backend Server running on port ${PORT}`);
  console.log(`📡 API endpoint: http://localhost:${PORT}`);
});
