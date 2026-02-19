# InfoSec-Project-Grp-1

# -LUCAS SECTION-
# To run, you need PostgreSQL installed and a database created: `CREATE DATABASE notevault;`.
# Set your DB credentials via the `DATABASE_URL` environment variable (recommended) or update the connection string in `backend/main.py`.
# Start the backend from the repository root so the `backend` package imports correctly:
#
# ```powershell
# cd "<repo-root>"
# python -m uvicorn backend.main:app --reload
# # or (with SSL):
# python -m uvicorn backend.main:app --ssl-certfile cert.pem --ssl-keyfile key.pem
# ```
#
# If you still prefer to run from inside the `backend` folder, use this alternative:
#
# ```powershell
# cd backend
# python -m uvicorn main:app --reload
# ```
#
# Key generation: the app will auto-generate and persist the master encryption key when the first user account is created.
# To pre-generate the key manually, run:
#
# ```powershell
# cd backend
# python init_crypto.py
# ```

# For AI Anomaly Detector: 
# Run to install dependencies: python -m pip install pandas scikit-learn joblib
### How It Works
- Logs user activity (login time, IP, device fingerprint, success/failure) to `user_activity` table.
- Trains an IsolationForest model on historical activity to detect anomalies.
- On successful login, checks if behavior is anomalous; if suspicious, forces security step-up (backup PIN change).

### Files
- `backend/activity.py` - Activity logging functions
- `backend/anomaly.py` - Model training and anomaly detection
- `backend/train_anomaly.py` - CLI script to retrain model
- `backend/seed_activity.py` - CLI script to generate test data

### Test Data Generation

Before training, generate realistic test data:

```bash
cd backend

# Generate 100 normal activity records
python seed_activity.py

# Generate 200 normal + 50 anomalous records
python seed_activity.py --count 200 --anomalies 50

# Clear existing data and generate fresh 100 records
python seed_activity.py --clear

# Full test dataset
python seed_activity.py --clear --count 300 --anomalies 100
```

This creates varied patterns:
- **Normal**: Business hours (8 AM-6 PM), familiar IPs, ~95% success rate
- **Anomalous**: Late night (2-4 AM), foreign IPs, ~60% failure rate

### Training & Retraining

#### First Time Setup
```bash
cd backend

# Seed test data
python seed_activity.py --clear --count 300 --anomalies 100

# Train model from activity logs
python train_anomaly.py --show-stats
```

This will:
1. Fetch all activity logs from the database
2. Train an IsolationForest model (contamination=0.05, i.e., 5% anomalies)
3. Save model to `backend/anomaly_model.pkl`
4. Display training statistics

#### Retraining (Manual)
```bash
python train_anomaly.py
```

#### Advanced Options
```bash
# Custom contamination rate (detect 10% of logins as anomalies)
python train_anomaly.py --contamination 0.1

# Save to custom location
python train_anomaly.py --output ./models/my_model.pkl

# Show training stats
python train_anomaly.py --show-stats
```

#### Scheduling (Automatic Retrain)

**Linux/macOS (Cron)**
```bash
# Edit crontab
crontab -e

# Add this line to retrain every Sunday at 2 AM
0 2 * * 0 cd /path/to/backend && python train_anomaly.py >> /var/log/notevault_retrain.log 2>&1
```

**Windows (Task Scheduler)**
1. Open Task Scheduler (tasksched.msc)
2. Create Basic Task → "NoteVault Anomaly Retrain"
3. Trigger: Weekly, Sunday 2:00 AM
4. Action: Start a program
   - Program: `C:\Python39\python.exe` (your Python path)
   - Arguments: `train_anomaly.py`
   - Start in: `C:\path\to\backend`

### Monitoring
- Check app startup logs: `✓ Anomaly model loaded` indicates success
- Failed logins are always logged regardless of model
- Suspicious login triggers email alert + forces backup PIN change




# -PRATHIP SECTION-
# pip install fastapi uvicorn sqlalchemy asyncpg python-multipart passlib[bcrypt] itsdangerous pyotp qrcode[pil] webauthn cryptography pyOpenSSL asn1crypto cbor2 numpy python-dotenv
please install the following packages in VS Code by pasting the above command in the terminal

For normal users: email verification (if new device login) -> TOTP (or backup pin if lost access to Google Auth)

For admin & superadmin users: email verification (if new device login) -> TOTP (or backup pin if lost access to Google Auth) -> Windows Passkey 

