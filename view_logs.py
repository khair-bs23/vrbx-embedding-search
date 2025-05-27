import time
import os

def view_logs():
    log_file = 'search_api.log'
    
    # Check if log file exists
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found. Make sure the application is running.")
        return
    
    print(f"Viewing logs from {log_file}. Press Ctrl+C to exit.")
    print("-" * 80)
    
    try:
        with open(log_file, 'r') as f:
            # Go to the end of the file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    print(line.strip())
                else:
                    time.sleep(0.1)  # Wait for new content
    except KeyboardInterrupt:
        print("\nStopped viewing logs.")

if __name__ == "__main__":
    view_logs() 