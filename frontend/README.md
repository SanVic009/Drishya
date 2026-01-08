# Drishya Frontend

Frontend application for the Drishya non-compliance detection system.

## Features

### Dashboard
- **Header Navigation**: Clean navigation bar with links to Overview, Analytics, Reports, and Settings
- **Collapsible Sidebar**: Toggle-able left sidebar with navigation options
- **System Status Indicator**: Real-time status display showing whether all systems are operational or if there are alerts
- **Camera Feed Grid**: 3-column responsive grid displaying multiple camera feeds
- **Modal Focus View**: Click on any camera feed to view it in full screen with a darkened background

### Design Principles
- Clean, professional design using shadcn/ui components
- No gradients, purple colors, or emojis
- Modern UI with Tailwind CSS
- Responsive layout that works on all screen sizes

## Technology Stack

- **React 18** with TypeScript
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - High-quality React components
- **Lucide React** - Beautiful icon set
- **Framer Motion** - For smooth animations

## Getting Started

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and visit: `http://localhost:5173`

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/
│   │   │   └── button.tsx         # Reusable button component
│   │   ├── Dashboard.tsx          # Main dashboard layout
│   │   ├── CameraGrid.tsx         # Camera feed grid with modal
│   │   └── WebcamFeed.tsx         # Component to access real webcam
│   ├── lib/
│   │   └── utils.ts               # Utility functions (cn helper)
│   ├── App.tsx                    # Root component
│   ├── main.tsx                   # Application entry point
│   └── index.css                  # Global styles with Tailwind
├── public/
├── index.html
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── vite.config.ts
```

## Components

### Dashboard
The main dashboard component includes:
- Fixed header with navigation and notifications
- Collapsible sidebar with menu items
- Status indicator (green checkmark for OK, red alert for issues)
- Main content area with camera grid

### Camera Grid
- 3-column responsive grid (adjusts to 2 columns on tablets, 1 on mobile)
- Each camera card shows:
  - Live thumbnail/image
  - Camera name and location
  - Live status indicator
  - Hover effect with zoom icon
- Click to open modal with focused view
- Modal includes:
  - Full-size camera feed
  - Close button
  - Camera details
  - Action buttons (Take Snapshot, Record)

## Customization

### Adding Real Camera Feeds
Replace the placeholder thumbnails in `CameraGrid.tsx` with your actual camera feed URLs:

```typescript
const cameraFeeds: CameraFeed[] = [
  {
    id: 1,
    name: 'Camera 01',
    thumbnail: 'YOUR_CAMERA_FEED_URL',
    location: 'Main Entrance',
    status: 'active',
  },
  // ... add more cameras
];
```

### Changing Status
Toggle between 'ok' and 'alert' status in `Dashboard.tsx`:

```typescript
const [systemStatus, setSystemStatus] = useState<'ok' | 'alert'>('alert');
```

## Next Steps

- Add landing page
- Integrate real camera feeds
- Add authentication
- Implement analytics page
- Add reports generation
- Set up alert notifications
- Add video recording functionality
- Implement AI-based non-compliance detection

## License

Private Project

import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
