"""
Drishya Detection System Startup Script
Starts all detection services with Redis-based video streaming
"""

import subprocess
import sys
import time
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

def log_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def log_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def log_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def log_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

def check_redis():
    """Check if Redis server is running"""
    log_info("Checking Redis server...")
    try:
        result = subprocess.run(
            ['redis-cli', 'ping'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and 'PONG' in result.stdout:
            log_success("Redis server is running")
            return True
        else:
            log_warning("Redis is not responding")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        log_error("Redis is not running")
        return False

def start_redis():
    """Start Redis server"""
    log_info("Starting Redis server...")
    try:
        subprocess.run(['sudo', 'systemctl', 'start', 'redis-server'], check=True)
        time.sleep(2)
        if check_redis():
            log_success("Redis server started successfully")
            return True
        else:
            log_error("Failed to start Redis")
            return False
    except subprocess.CalledProcessError:
        log_error("Failed to start Redis server")
        return False

def start_service(name, port, script):
    """Start a detection service in background"""
    log_info(f"Starting {name} on port {port}...")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Open log file
    log_file = logs_dir / f"{name}.log"
    pid_file = logs_dir / f"{name}.pid"
    
    with open(log_file, 'w') as log_f:
        # Start process
        process = subprocess.Popen(
            ['conda', 'run', '-n', 'gen', '--cwd', 
             '/home/sanvict/Documents/Code/Drishya/backend', 
             'python', script],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        
        # Save PID
        with open(pid_file, 'w') as pid_f:
            pid_f.write(str(process.pid))
        
        # Wait a bit and check if process is still running
        time.sleep(2)
        
        if process.poll() is None:
            log_success(f"{name} started (PID: {process.pid})")
            return True
        else:
            log_error(f"Failed to start {name}")
            return False

def main():
    """Main entry point"""
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print(f"{Colors.BLUE}   Drishya Multi-Model Detection System{Colors.NC}")
    print(f"{Colors.BLUE}{'='*50}{Colors.NC}")
    print()
    
    # Check/start Redis
    if not check_redis():
        if not start_redis():
            log_error("Cannot proceed without Redis. Exiting.")
            sys.exit(1)
    
    print()
    log_info("Starting camera publisher...")
    if not start_service("camera_publisher", "5004", "camera_publisher.py"):
        log_error("Failed to start camera publisher")
        sys.exit(1)
    
    print()
    log_info("Starting detection services...")
    
    # Start Anti-Cheat Detector (port 5002)
    if not start_service("anticheat_detector", "5002", "anticheat_detector.py"):
        log_warning("Anti-cheat detector failed to start")
    
    # Start QR Code Detector (port 5003)
    if not start_service("qr_detector", "5003", "qr_detector.py"):
        log_warning("QR detector failed to start")
    
    # Start Anomaly Detector (port 5001)
    if not start_service("anomaly_detector", "5001", "anomaly_detector.py"):
        log_warning("Anomaly detector failed to start")
    
    print()
    print(f"{Colors.GREEN}{'='*55}{Colors.NC}")
    print(f"{Colors.GREEN}  All detection services started successfully!{Colors.NC}")
    print(f"{Colors.GREEN}{'='*55}{Colors.NC}")
    print()
    
    print(f"{Colors.BLUE}Service Endpoints:{Colors.NC}")
    print("  - Camera Feed: http://localhost:5004/api/raw_feed")
    print("  - Anti-Cheat:  http://localhost:5002/api/anticheat_feed")
    print("  - QR Code:     http://localhost:5003/api/qr_feed")
    print("  - Anomaly:     http://localhost:5001/api/video_feed")
    print()
    
    print(f"{Colors.BLUE}Stats Endpoints:{Colors.NC}")
    print("  - Anti-Cheat:  http://localhost:5002/api/anticheat_stats")
    print("  - QR Code:     http://localhost:5003/api/qr_stats")
    print("  - Anomaly:     http://localhost:5001/api/anomaly_stats")
    print()
    
    print(f"{Colors.YELLOW}Logs are being written to: ./logs/{Colors.NC}")
    print()
    print(f"{Colors.YELLOW}To stop all services, run: conda run -n gen python stop_services.py{Colors.NC}")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStartup interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        sys.exit(1)
