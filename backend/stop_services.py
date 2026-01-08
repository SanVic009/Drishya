"""
Drishya Detection System Stop Script
Stops all detection services
"""

import sys
import os
import signal
import subprocess
from pathlib import Path

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

def log_info(msg):
    print(f"{Colors.YELLOW}[INFO]{Colors.NC} {msg}")

def log_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def log_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def stop_service_by_name(service_name):
    """Stop a service by finding and killing processes by name"""
    try:
        # Find all Python processes matching the service name
        cmd = f'ps aux | grep -E "({service_name})" | grep python | grep -v grep | awk \'{{print $2}}\''
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        pids = result.stdout.strip().split('\n')
        pids = [pid for pid in pids if pid]  # Remove empty strings
        
        if not pids:
            log_warning(f"No running process found for {service_name}")
            return
        
        log_info(f"Found {len(pids)} process(es) for {service_name}")
        
        for pid in pids:
            try:
                pid_int = int(pid)
                log_info(f"Stopping {service_name} (PID: {pid_int})...")
                os.kill(pid_int, signal.SIGTERM)
                
                # Wait a bit
                import time
                time.sleep(0.5)
                
                # Check if still running, force kill if needed
                try:
                    os.kill(pid_int, 0)
                    log_warning(f"Force killing {service_name} (PID: {pid_int})...")
                    os.kill(pid_int, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already terminated
                    
                log_success(f"{service_name} (PID: {pid_int}) stopped")
                
            except (ValueError, ProcessLookupError) as e:
                log_warning(f"Could not stop process {pid}: {e}")
                
    except Exception as e:
        log_warning(f"Error stopping {service_name}: {e}")

def cleanup_pid_files():
    """Clean up all PID files"""
    log_info("Cleaning up PID files...")
    pid_dir = Path("logs")
    if pid_dir.exists():
        for pid_file in pid_dir.glob("*.pid"):
            try:
                pid_file.unlink()
                log_info(f"Removed {pid_file.name}")
            except Exception as e:
                log_warning(f"Could not remove {pid_file.name}: {e}")

def main():
    """Main entry point"""
    print(f"{Colors.YELLOW}Stopping Drishya Detection Services...{Colors.NC}")
    print()
    
    # Stop all services by process name
    stop_service_by_name("camera_publisher")
    stop_service_by_name("anticheat_detector")
    stop_service_by_name("qr_detector")
    stop_service_by_name("anomaly_detector")
    
    print()
    
    # Clean up PID files
    cleanup_pid_files()
    
    print()
    log_success("All services stopped")
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}[ERROR]{Colors.NC} Unexpected error: {e}")
        sys.exit(1)
