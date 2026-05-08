# Attendance Management System

A Flask-based attendance management system with real-time face recognition using YOLO and OpenCV. The system captures photos, detects faces, and sends email alerts for attendance events.

## Features

- **Face Recognition**: Uses YOLO v8 and OpenCV for real-time face detection
- **Attendance Tracking**: Logs attendance in MySQL database
- **Email Alerts**: Sends email notifications for attendance events
- **User Authentication**: Secure login system
- **Web Interface**: Flask-based web application
- **Photo Storage**: Automatic capture and storage of attendance photos

## Prerequisites

- Python 3.8+
- MySQL Server
- OpenCV dependencies (for video/camera access)
- Virtual environment (recommended)

## Installation

### Quick Setup (One Command)

For easy setup on any computer, simply run:

```bash
chmod +x setup.sh
./setup.sh
```

This will automatically:
- Check Python installation
- Create virtual environment (.venv)
- Install all dependencies
- Create required directories

Then manually update `app.py` with your database and email credentials.

### Manual Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/ysathyasai/Attendance-Management-System
cd Attendance-Management-System
```

#### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install flask werkzeug opencv-python numpy pymysql pillow
pip install ultralytics supervision  # For YOLO face detection (optional but recommended)
```

#### 4. Configure Database

#### Create MySQL Database

```sql
CREATE DATABASE attendance_management;
USE attendance_management;

-- Create tables as needed (adjust schema based on app.py requirements)
```

#### Update Database Credentials

Edit `app.py` and update the `DB_CONFIG` dictionary with your MySQL credentials:

```python
DB_CONFIG = dict(
    host       = "localhost",
    user       = "root",
    password   = "your_password_here",
    database   = "attendance_management",
)
```

#### 5. Configure Email Settings

Update the email configuration in `app.py`:

```python
EMAIL_FROM = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"  # Use Gmail App Password if 2FA enabled
```

#### 6. Configure Application Settings

Update the following settings in `app.py`:

```python
USERNAME = "your_username"
PASSWORD = "your_password"
NGROK_LINK = "your_ngrok_url"  # If using ngrok for tunneling
```

#### 7. Prepare Required Files

Ensure the following files are in the project root:

- `classifier.xml` - Haar Cascade classifier
- `label_map.json` - Label mapping for classification
- `alarm.wav` - Alarm audio file
- `haarcascade_frontalface_default.xml` - OpenCV Haar Cascade
- `yolo26.pt` - YOLO model weights

## Running the Application

### 1. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 2. Start the Flask Application

```bash
python app.py
```

The application will start on `http://localhost:5000` (or the configured port)

### 3. Access the Web Interface

Open your browser and navigate to:
```
http://localhost:5000
```

#### Login Credentials

Use the credentials configured in the `USERNAME` and `PASSWORD` variables in `app.py`.

## Project Structure

```
ysathyasai/
├── app.py                          # Main Flask application
├── templates/
│   ├── index.html                 # Dashboard page
│   └── login.html                 # Login page
├── alerts/                         # Alert storage directory
├── photo samples/                  # Photo capture storage
├── classifier.xml                  # ML classifier model
├── label_map.json                  # Label mapping
├── alarm.wav                       # Alert sound
├── haarcascade_frontalface_default.xml  # Haar Cascade
├── yolo26.pt                       # YOLO model
├── cloudflared                     # Cloudflare tunnel config (optional)
└── .venv/                          # Virtual environment
```

## Troubleshooting

### Virtual Environment Activation Issues

If `source .venv/bin/activate` returns Exit Code 1:

```bash
# Rebuild virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

### Missing Dependencies

If you encounter import errors, reinstall dependencies:

```bash
pip install --upgrade -r requirements.txt
# Or manually install:
pip install flask werkzeug opencv-python numpy pymysql pillow ultralytics supervision
```

### Database Connection Issues

- Ensure MySQL is running: `sudo service mysql status`
- Check credentials in `DB_CONFIG`
- Verify database exists: `mysql -u root -p -e "SHOW DATABASES;"`

### Camera/Video Issues

Ensure you have the necessary OpenCV dependencies:

```bash
# On Ubuntu/Debian
sudo apt-get install python3-opencv libsm6 libxext6 libxrender-dev
```

### YOLO Module Not Found

The application will work without YOLO but face detection features may be disabled:

```bash
pip install ultralytics supervision
# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Optional: Using ngrok for Tunneling

To expose your local application to the internet:

```bash
# Install ngrok
curl https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.zip -o ngrok.zip
unzip ngrok.zip

# Run ngrok on port 5000
./ngrok http 5000

# Update NGROK_LINK in app.py with the generated URL
```

## Development Tips

- Keep the `.venv` directory for consistent development environment
- Always activate the virtual environment before running the application
- Use environment variables for sensitive credentials (consider using `.env` file)
- Test database connectivity before starting the application

## Support

For issues or questions, check:
1. The error logs in terminal output
2. MySQL error logs
3. Browser console for frontend issues
